"""api 组：历史列表上限/全集搜索（ADR-0126）与 历史分组实体（ADR-0129）。

钉住的口径：

- 列表不再静默截断（旧写死 [:50]）：默认 500，回 total/matched/truncated；搜索走服务端
  全集，不受 limit 影响；
- 分组是**实体**：`results/.history-groups.json` = `{v:2, groups:[{id,name,note}],
  assign:{run_id: group_id}}`；组 id 服务端生成，组名只是它的一个属性（改错字不再变出第二个组）；
- 旧平铺 `{run_id: 组名}` 读到就在内存折算，**不写盘**（改用户盘要另开一条写路径去防）；
- 写入口只有一个 `POST /api/bench/groups`（body `{ops:[create|move|rename|merge|dissolve]}`），
  整笔原子：任一 op 非法 → 400 且**一个字节都不写**（半截归组比不做更难解释）；
- 组名/组 id 不进 (mtime,size) 摘要缓存（改组后列表必须立刻变）；`.history-groups.json`
  不被当成一条历史记录；点开头 → run_id 路由天然不可达；
- **动组不改任何 `results/*.json` 的字节**（`test_group_ops_never_touch_archive_bytes`
  是防"分组又缝回存档里"的门禁断言，不许删）；
- 组名/备注超长**报错不截断**；「未分组」是保留名；已删存档的孤儿 assign 读时过滤不报错。

    python -m pytest tests/api/test_history_groups.py -q
"""
import hashlib
import json
import os
import tempfile
import unittest

from fastapi.testclient import TestClient

import server


MAP = ".history-groups.json"


def _archive(rid, note="", models=("m-a",), results=None):
    d = {"run_id": rid, "started_at": "2026-10-10 09:00:00",
         "cfg": {"provider": "p1", "models": list(models), "scenarios": ["code"],
                 "note": note},
         "results": results if results is not None else
         [{"model": models[0], "scenario": "code", "all_ok": True}]}
    with open(os.path.join(server.RESULTS_DIR, f"{rid}.json"), "w",
              encoding="utf-8") as f:
        json.dump(d, f)
    return rid


def _fingerprint(exclude_map=True):
    """目录里每个文件的 (字节 sha1, mtime_ns)：断言"一个字都没动"用，比只看 size 硬。"""
    out = {}
    for name in sorted(os.listdir(server.RESULTS_DIR)):
        p = os.path.join(server.RESULTS_DIR, name)
        if exclude_map and name == MAP:
            continue
        st = os.stat(p)
        with open(p, "rb") as f:
            out[name] = (hashlib.sha1(f.read()).hexdigest(), st.st_mtime_ns)
    return out


class TestHistoryList(unittest.TestCase):
    """列表上限、总数披露、全集搜索（ADR-0126，与分组共用这份列表）。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._results = server.RESULTS_DIR
        server.RESULTS_DIR = self.tmp.name
        os.makedirs(server.RESULTS_DIR, exist_ok=True)
        server._HIST_CACHE.clear()
        self.client = TestClient(server.app)
        for i in range(7):
            _archive(f"r-{i}", note=f"note{i}")

    def tearDown(self):
        server.RESULTS_DIR = self._results
        server._HIST_CACHE.clear()
        self.tmp.cleanup()

    def test_no_silent_truncation(self):
        d = self.client.get("/api/bench/history").json()
        self.assertEqual(d["total"], 7)
        self.assertEqual(d["matched"], 7)
        self.assertFalse(d["truncated"])
        self.assertEqual(len(d["history"]), 7)

    def test_limit_reports_total_and_truncated(self):
        d = self.client.get("/api/bench/history?limit=3").json()
        self.assertEqual(len(d["history"]), 3)
        self.assertEqual(d["total"], 7)
        self.assertTrue(d["truncated"])

    def test_search_covers_records_beyond_limit(self):
        # 最旧那条（r-0）在 limit=1 的默认视图里看不见，但搜索必须能命中它
        d = self.client.get("/api/bench/history?limit=1&q=note0").json()
        self.assertEqual(d["matched"], 1)
        self.assertEqual([h["run_id"] for h in d["history"]], ["r-0"])

    def test_search_matches_model_and_run_id(self):
        for q in ("m-a", "r-3"):
            self.assertEqual(
                self.client.get(f"/api/bench/history?q={q}").json()["matched"],
                7 if q == "m-a" else 1, q)

    def test_summary_new_fields(self):
        _archive("r-bad", results=[{"model": "m-a", "all_ok": False},
                                   {"model": "m-a", "all_ok": True}])
        h = {x["run_id"]: x for x in
             self.client.get("/api/bench/history").json()["history"]}["r-bad"]
        self.assertEqual(h["n_failed"], 1)
        self.assertEqual(h["provider"], "p1")
        self.assertGreater(h["size"], 0)
        self.assertIn("mtime", h)

    def test_cache_eviction_keeps_other_entries(self):
        """超上限逐条淘汰，不再整体清空（条目多时每次请求全量重解析 = ADR-0082 反例）。"""
        touched = self.client.get("/api/bench/history").json()["history"]
        self.assertTrue(touched)
        _archive("r-new", note="新的一条")   # 必须是个没进缓存的新档，否则走命中分支不触发淘汰
        saved = server._HIST_CACHE_MAX
        server._HIST_CACHE_MAX = 2
        try:
            self.assertTrue(server._archive_summary(
                os.path.join(server.RESULTS_DIR, "r-new.json"), "r-new.json"))
        finally:
            server._HIST_CACHE_MAX = saved
        self.assertEqual(len(server._HIST_CACHE), 2)   # 收敛到上限，而不是停在超限
        self.assertIn("r-new.json", server._HIST_CACHE)   # 刚碰过的必须还在


class TestHistoryGroupsEntity(unittest.TestCase):
    """分组实体：GET 形状 + `POST /api/bench/groups` 的五个 op 与整笔原子。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._results = server.RESULTS_DIR
        server.RESULTS_DIR = self.tmp.name
        os.makedirs(server.RESULTS_DIR, exist_ok=True)
        server._HIST_CACHE.clear()
        self.client = TestClient(server.app)
        for rid in ("g-a1", "g-a2", "g-b1"):
            _archive(rid)

    def tearDown(self):
        server.RESULTS_DIR = self._results
        server._HIST_CACHE.clear()
        self.tmp.cleanup()

    # ---- 端点/文件 helper（形状断言全部走 GET，与前端读的是同一份） ---------------

    def _post(self, ops):
        return self.client.post("/api/bench/groups", json={"ops": ops})

    def _get(self):
        return self.client.get("/api/bench/history").json()

    def _glist(self):
        return self._get()["groups"]

    def _byname(self, name):
        for g in self._glist():
            if g["name"] == name:
                return g
        return None

    def _summary(self, rid):
        return {h["run_id"]: h for h in self._get()["history"]}[rid]

    def _map_raw(self):
        with open(os.path.join(server.RESULTS_DIR, MAP), encoding="utf-8") as f:
            return json.load(f)

    def _write_map(self, d):
        with open(os.path.join(server.RESULTS_DIR, MAP), "w", encoding="utf-8") as f:
            json.dump(d, f)

    def _create(self, name, note=None, run_ids=None):
        op = {"op": "create", "name": name}
        if note is not None:
            op["note"] = note
        if run_ids is not None:
            op["run_ids"] = run_ids
        r = self._post([op])
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    # ---- GET 的新字段位置 --------------------------------------------------------

    def test_get_shape_groups_and_summary_fields(self):
        """前端照这份写：顶层 groups=[{id,name,note,count}]，每条摘要 group + group_id。"""
        self._create("4090 横评", note="本地 nvfp4+fp8kv", run_ids=["g-a1", "g-a2"])
        d = self._get()
        self.assertEqual(len(d["groups"]), 1)
        g = d["groups"][0]
        self.assertEqual(sorted(g.keys()), ["count", "id", "name", "note"])
        self.assertEqual((g["name"], g["note"], g["count"]), ("4090 横评", "本地 nvfp4+fp8kv", 2))
        self.assertTrue(g["id"].startswith("g-"))
        a1, a2, b1 = self._summary("g-a1"), self._summary("g-a2"), self._summary("g-b1")
        self.assertEqual((a1["group"], a1["group_id"]), (g["name"], g["id"]))
        self.assertEqual(a2["group_id"], g["id"])
        self.assertEqual((b1["group"], b1["group_id"]), ("", ""))   # 未分组 = 两个空串

    def test_groups_file_is_v2_after_write(self):
        self._create("A", run_ids=["g-a1"])
        raw = self._map_raw()
        self.assertEqual(raw["v"], 2)
        self.assertEqual(raw["groups"], [{"id": raw["groups"][0]["id"],
                                          "name": "A", "note": ""}])
        self.assertEqual(list(raw["assign"]), ["g-a1"])
        self.assertEqual(raw["assign"]["g-a1"], raw["groups"][0]["id"])

    def test_empty_group_is_creatable_and_listed(self):
        """ADR-0129 的第 1 号动机：建组必须能不带记录，且建完要看得见（count=0）。"""
        r = self._create("下周要跑的")
        self.assertEqual(len(r["created"]), 1)
        self.assertEqual(r["touched"], 0)
        g = self._byname("下周要跑的")
        self.assertIsNotNone(g)
        self.assertEqual(g["count"], 0)
        self.assertTrue(all(h["group"] == "" for h in self._get()["history"]))

    def test_groups_ledger_ignores_limit_and_query(self):
        """groups 是组台账，不是当前视图统计：limit/q 不得改变它。"""
        self._create("A", run_ids=["g-a1"])
        before = self._glist()
        self.assertEqual(self.client.get("/api/bench/history?limit=1").json()["groups"], before)
        self.assertEqual(self.client.get("/api/bench/history?q=zzz").json()["groups"], before)

    # ---- 五种 op ---------------------------------------------------------------

    def test_op_create_moves_given_records(self):
        r = self._create("A", run_ids=["g-a1", "g-a2"])
        self.assertEqual(r["touched"], 2)
        self.assertEqual(self._byname("A")["count"], 2)

    def test_op_move_into_another_group(self):
        a = self._create("A", run_ids=["g-a1"])["created"][0]
        b = self._create("B")["created"][0]
        r = self._post([{"op": "move", "id": b, "run_ids": ["g-a1"]}])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["touched"], 1)
        self.assertEqual(self._summary("g-a1")["group"], "B")
        self.assertEqual(self._byname("A")["count"], 0)   # 空组留在台账里，不自动消失
        self.assertEqual(self._byname("B")["count"], 1)

    def test_op_move_with_empty_id_is_move_out(self):
        a = self._create("A", run_ids=["g-a1", "g-a2"])["created"][0]
        r = self._post([{"op": "move", "id": "", "run_ids": ["g-a1"]}])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["touched"], 1)
        self.assertEqual((self._summary("g-a1")["group"],
                          self._summary("g-a1")["group_id"]), ("", ""))
        self.assertEqual(self._summary("g-a2")["group"], "A")   # 同组别条不动
        self.assertNotIn("g-a1", self._map_raw()["assign"])     # 移出 = 删键，不写空串

    def test_op_rename_then_list_changes_at_once(self):
        """改名后列表立刻变 = 组名没被烤进 (mtime,size) 摘要缓存；且存档字节不动。"""
        a = self._create("第一批", run_ids=["g-a1"])["created"][0]
        fp = _fingerprint()
        r = self._post([{"op": "rename", "id": a, "name": "第二批", "note": "含 iq3"}])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["touched"], 0)   # 改组自身不碰任何记录
        self.assertEqual(self._summary("g-a1")["group"], "第二批")
        self.assertEqual(self._byname("第二批")["note"], "含 iq3")
        self.assertEqual(self._byname("第一批"), None)
        self.assertEqual(_fingerprint(), fp)

    def test_op_rename_note_only_keeps_name(self):
        a = self._create("A", run_ids=["g-a1"], note="旧备注")["created"][0]
        self._post([{"op": "rename", "id": a, "note": "新备注"}])
        self.assertEqual((self._byname("A")["name"], self._byname("A")["note"]),
                         ("A", "新备注"))
        self._post([{"op": "rename", "id": a, "note": ""}])   # 空串 = 清空备注
        self.assertEqual(self._byname("A")["note"], "")

    def test_op_merge_whole_group(self):
        a = self._create("A", run_ids=["g-a1", "g-a2"])["created"][0]
        b = self._create("B", run_ids=["g-b1"])["created"][0]
        r = self._post([{"op": "merge", "id": a, "into": b}])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["dissolved"], [a])
        self.assertEqual([g["name"] for g in self._glist()], ["B"])   # 被并的组消失
        self.assertEqual(self._byname("B")["count"], 3)
        for rid in ("g-a1", "g-a2"):
            self.assertEqual(self._summary(rid)["group"], "B")
        self.assertNotIn(a, self._map_raw()["assign"].values())

    def test_op_dissolve_returns_records_to_ungrouped(self):
        a = self._create("A", run_ids=["g-a1", "g-a2"])["created"][0]
        r = self._post([{"op": "dissolve", "id": a}])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["dissolved"], [a])
        self.assertEqual(self._glist(), [])
        self.assertEqual([self._summary(x)["group"] for x in ("g-a1", "g-a2")], ["", ""])
        self.assertNotIn(a, self._map_raw()["assign"].values())

    def test_multiple_ops_in_one_request_are_one_transaction(self):
        """一次请求做完整笔：拆成多次 = 并发重写同一份映射，后写的覆盖先写的。"""
        r = self._post([{"op": "create", "name": "A", "run_ids": ["g-a1"]},
                        {"op": "create", "name": "B", "run_ids": ["g-a2", "g-b1"]}])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(len(r.json()["created"]), 2)
        self.assertEqual([g["name"] for g in self._glist()], ["A", "B"])
        self.assertEqual(self._byname("B")["count"], 2)

    # ---- create 的同名转移入 / 保留名 / 上限 -------------------------------------

    def test_create_with_existing_name_moves_into_it(self):
        """保留"输已有组名即并入"的老便利：不新建第二个组、不产生同名对。"""
        first = self._create("A", note="甲", run_ids=["g-a1"])
        r = self._create("A", note="乙", run_ids=["g-a2"])
        self.assertEqual(r["created"], [], "撞同名不得新建组")
        self.assertEqual(r["touched"], 1)
        self.assertEqual(len(self._glist()), 1)
        self.assertEqual(self._byname("A")["count"], 2)
        self.assertEqual(self._byname("A")["note"], "甲")   # 不悄悄改已有组的备注（改备注用 rename）
        self.assertEqual(self._byname("A")["id"], first["created"][0])
        self.assertEqual(self._summary("g-a2")["group"], "A")

    def test_create_with_trimmed_equal_name_is_same_group(self):
        self._create("A", run_ids=["g-a1"])
        r = self._create("  A  ", run_ids=["g-a2"])
        self.assertEqual(r["created"], [])
        self.assertEqual(len(self._glist()), 1)

    def test_reserved_name_ungrouped_rejected(self):
        """「未分组」是列表给未归组记录的伪组名，当真组名会出现两个答案。"""
        r = self._post([{"op": "create", "name": server.GROUP_RESERVED_NAME}])
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self._glist(), [])
        a = self._create("A")["created"][0]
        self.assertEqual(self._post(
            [{"op": "rename", "id": a, "name": server.GROUP_RESERVED_NAME}]).status_code, 400)

    def test_rename_into_another_group_name_rejected(self):
        self._create("A", run_ids=["g-a1"])
        b = self._create("B")["created"][0]
        r = self._post([{"op": "rename", "id": b, "name": "A"}])
        self.assertEqual(r.status_code, 400)
        self.assertEqual(sorted(g["name"] for g in self._glist()), ["A", "B"])

    def test_rename_to_own_name_allowed(self):
        a = self._create("A", run_ids=["g-a1"])["created"][0]
        self.assertEqual(self._post([{"op": "rename", "id": a, "name": "A"}]).status_code, 200)

    def test_group_count_cap(self):
        seed = {"v": 2,
                "groups": [{"id": f"g-i{i}", "name": f"G{i}", "note": ""} for i in range(200)],
                "assign": {}}
        self._write_map(seed)
        r = self._post([{"op": "create", "name": "第 201 个"}])
        self.assertEqual(r.status_code, 400)
        self.assertEqual(len(self._glist()), 200)

    def test_touched_cap(self):
        saved = server.GROUP_ASSIGN_MAX_IDS
        server.GROUP_ASSIGN_MAX_IDS = 1
        try:
            self.assertEqual(self._post([{"op": "create", "name": "A",
                                          "run_ids": ["g-a1", "g-a2"]}]).status_code, 400)
        finally:
            server.GROUP_ASSIGN_MAX_IDS = saved
        self.assertEqual(self._glist(), [])

    def test_touched_counts_effective_changes_only(self):
        """touched 数“有效归属真的变了”：移去它本来就在的组不算动过，不给确认框报虚数。"""
        a = self._create("A", run_ids=["g-a1"])["created"][0]
        self.assertEqual(self._post([{"op": "move", "id": a,
                                      "run_ids": ["g-a1"]}]).json()["touched"], 0)

    # ---- 校验不过 = 整笔不做 ------------------------------------------------------

    def test_same_record_in_two_groups_rejected_atomic(self):
        """同一条进两个组 = 服务端替用户静默选了一个答案；且整笔一个字不写。"""
        a = self._create("A", run_ids=["g-a1"])["created"][0]
        b = self._create("B")["created"][0]
        fp_all = _fingerprint()
        map_path = os.path.join(server.RESULTS_DIR, MAP)
        with open(map_path, "rb") as f:
            before_bytes = f.read()
        before_ns = os.stat(map_path).st_mtime_ns
        r = self._post([{"op": "move", "id": a, "run_ids": ["g-a2"]},
                        {"op": "move", "id": b, "run_ids": ["g-a2"]}])
        self.assertEqual(r.status_code, 400)
        with open(map_path, "rb") as f:
            self.assertEqual(f.read(), before_bytes, "被拒的请求不得改映射文件一个字节")
        self.assertEqual(os.stat(map_path).st_mtime_ns, before_ns, "被拒的请求不得重写映射文件")
        self.assertEqual(_fingerprint(), fp_all)
        self.assertEqual(self._summary("g-a2")["group"], "")   # 不是"做了一半"

    def test_rejected_request_does_not_create_map_file(self):
        fp = _fingerprint()
        for body in ({"ops": []}, {"ops": "x"}, {}, {"ops": [{"op": "create"}]},
                     {"ops": [{"op": "nope", "id": "g-aaaaaa"}]}):
            self.assertEqual(self.client.post("/api/bench/groups", json=body).status_code,
                             400, f"应拒: {body}")
        self.assertFalse(os.path.exists(os.path.join(server.RESULTS_DIR, MAP)))
        self.assertEqual(_fingerprint(), fp)

    def test_op_validation(self):
        a = self._create("A", run_ids=["g-a1"])["created"][0]
        fp = _fingerprint()
        map_path = os.path.join(server.RESULTS_DIR, MAP)
        with open(map_path, "rb") as f:
            before_bytes = f.read()
        bad = [
            {"op": "create", "name": ""},                       # 空组名
            {"op": "create", "name": 3},                        # 非串组名
            {"op": "create", "name": "B", "run_ids": "g-a1"},   # run_ids 不是数组
            {"op": "create", "name": "B", "note": 1},           # note 非串
            {"op": "move"},                                     # 缺 id/run_ids
            {"op": "move", "id": a},                            # 缺 run_ids
            {"op": "move", "id": None, "run_ids": []},          # id 非串
            {"op": "move", "id": "g-ghost", "run_ids": []},     # 组不存在
            {"op": "rename", "id": a},                          # name/note 都没给
            {"op": "rename", "id": "g-ghost", "name": "X"},     # 组不存在
            {"op": "rename", "id": a, "name": ""},              # 空组名
            {"op": "merge", "id": a},                           # 缺 into
            {"op": "merge", "id": a, "into": "g-ghost"},        # into 不存在
            {"op": "merge", "id": a, "into": a},                # 并给自己
            {"op": "dissolve"},                                 # 缺 id
            {"op": "dissolve", "id": "g-ghost"},                # 组不存在
            "不是对象",
        ]
        for op in bad:
            r = self._post([op] if isinstance(op, dict) else [op])
            self.assertEqual(r.status_code, 400, f"应拒: {op}")
        with open(map_path, "rb") as f:
            self.assertEqual(f.read(), before_bytes)
        self.assertEqual(_fingerprint(), fp)
        self.assertEqual(self._summary("g-a1")["group"], "A")   # 什么都没动

    def test_unknown_group_id_rejected_twice_dissolved(self):
        a = self._create("A", run_ids=["g-a1"])["created"][0]
        self.assertEqual(self._post([{"op": "dissolve", "id": a}]).status_code, 200)
        r = self._post([{"op": "dissolve", "id": a}])   # 老标签页拿着退役 id 再点一次
        self.assertEqual(r.status_code, 400)
        self.assertIn("组不存在", r.json()["detail"])

    def test_run_id_validation_rejects_traversal(self):
        a = self._create("A", run_ids=["g-a1"])["created"][0]
        fp = _fingerprint()
        for rid in ("a/../b", "a..b", "../g-a1", "x" * 65, "g a1", "g-a1.json"):
            for bad in ({"op": "create", "name": "B", "run_ids": [rid]},
                        {"op": "move", "id": a, "run_ids": [rid]},
                        {"op": "move", "id": "", "run_ids": [rid]}):
                self.assertEqual(self._post([bad]).status_code, 400, f"应拒: {bad}")
        self.assertEqual(self._post([{"op": "move", "id": a, "run_ids": [123]}]).status_code, 400)
        self.assertEqual(_fingerprint(), fp)
        self.assertEqual(len(self._glist()), 1)   # 没有多出半个组

    def test_overlong_name_and_note_rejected_not_truncated(self):
        """超长**报错**而不静默截断：截断后的名字与用户心里那个不是同一个，以后按名合并合不上。"""
        long_name = "名" * 60
        long_note = "备" * 80
        for op in ({"op": "create", "name": long_name},
                   {"op": "create", "name": "A", "note": long_note},
                   {"op": "create", "name": long_name, "run_ids": ["g-a1"]}):
            r = self._post([op])
            self.assertEqual(r.status_code, 400, f"应拒: {op}")
            self.assertIn("字符", r.json()["detail"])
        self.assertEqual(self._glist(), [])
        self.assertFalse(os.path.exists(os.path.join(server.RESULTS_DIR, MAP)),
                         "超长不截断：截半的名字根本不该出现在盘上")
        a = self._create("A", note="x" * 60)["created"][0]
        self.assertEqual(self._post([{"op": "rename", "id": a,
                                      "name": long_name}]).status_code, 400)
        self.assertEqual(self._post([{"op": "rename", "id": a,
                                      "note": long_note}]).status_code, 400)
        self.assertEqual((self._byname("A")["name"], self._byname("A")["note"]),
                         ("A", "x" * 60))   # 原样保留，没被截

    def test_boundary_lengths_accepted(self):
        """上限本身可用：40 字组名 / 60 字备注是合法值（报错只针对越界）。"""
        name, note = "名" * server.GROUP_NAME_MAX, "备" * server.GROUP_NOTE_MAX
        r = self._create(name, note=note, run_ids=["g-a1"])
        self.assertEqual(len(r["created"]), 1)
        self.assertEqual((self._byname(name)["name"], self._byname(name)["note"]), (name, note))

    def test_group_name_whitespace_trimmed_in_storage(self):
        self._create("  夜间跑批  ", run_ids=["g-a1"])
        self.assertEqual(self._byname("夜间跑批")["name"], "夜间跑批")
        self.assertEqual(self._summary("g-a1")["group"], "夜间跑批")

    # ---- 旧形状折算（不写盘） ------------------------------------------------------

    def test_legacy_flat_file_folds_without_writing(self):
        legacy = {"g-a1": "旧甲", "g-a2": "旧甲", "g-b1": "旧乙"}
        self._write_map(legacy)
        before = _fingerprint()
        d = self._get()
        self.assertEqual([g["name"] for g in d["groups"]], ["旧甲", "旧乙"])   # 名字首次出现序
        self.assertEqual([g["count"] for g in d["groups"]], [2, 1])
        self.assertTrue(all(g["note"] == "" for g in d["groups"]))
        s = {h["run_id"]: h for h in d["history"]}
        self.assertEqual((s["g-a1"]["group"], s["g-b1"]["group"]), ("旧甲", "旧乙"))
        self.assertTrue(s["g-a1"]["group_id"].startswith("g-"))
        self.assertEqual(_fingerprint(), before)   # 折算只在内存：不许改用户的盘
        with open(os.path.join(server.RESULTS_DIR, MAP), encoding="utf-8") as f:
            self.assertEqual(json.load(f), legacy)

    def test_legacy_fold_ids_are_stable_across_reads(self):
        """随机 id 会让同一个组名每次换一个 id，前端刚拿到的 id 下一步就 400。"""
        self._write_map({"g-a1": "旧甲", "g-b1": "旧乙"})
        first = self._glist()
        self.assertEqual([g["id"] for g in self._glist()], [g["id"] for g in first])
        self.assertEqual(self._post([{"op": "rename", "id": first[0]["id"],
                                      "name": "新甲"}]).status_code, 200)
        self.assertEqual(sorted(g["name"] for g in self._glist()), ["新甲", "旧乙"])
        self.assertEqual(self._map_raw()["v"], 2)   # 第一次写自然落成 v2

    def test_legacy_junk_entries_fold_as_ungrouped(self):
        self._write_map({"g-a1": "旧甲", "g-a2": "   ", "g-b1": 7, "v2keep": "旧甲"})
        d = self._get()
        self.assertEqual([g["name"] for g in d["groups"]], ["旧甲"])
        self.assertEqual([g["count"] for g in d["groups"]], [1])
        self.assertEqual(self._summary("g-a2")["group"], "")

    # ---- 孤儿 / 损坏 / 边界文件 -----------------------------------------------------

    def test_orphan_assign_and_missing_group_are_silent(self):
        a = self._create("A", run_ids=["g-a1"])["created"][0]
        self.client.delete("/api/bench/history/g-a1")   # 删档 → 映射里留孤儿
        raw = self._map_raw()
        raw["assign"]["g-ghost-run"] = a          # 指向不存在存档的条目
        raw["assign"]["g-b1"] = "g-gone"          # 指向失踪组的条目 = 未分组
        self._write_map(raw)
        d = self._get()
        self.assertEqual(d["total"], 2)                        # 不报错、不冒幽灵条目
        self.assertEqual([g["id"] for g in d["groups"]], [a])  # 失踪组不进台账
        self.assertEqual([g["count"] for g in d["groups"]], [0])   # 存档没了 → 空组形状
        self.assertEqual((self._summary("g-b1")["group"],
                          self._summary("g-b1")["group_id"]), ("", ""))

    def test_corrupt_map_degrades_to_no_groups(self):
        with open(os.path.join(server.RESULTS_DIR, MAP), "w", encoding="utf-8") as f:
            f.write("{ 半截")
        d = self._get()
        self.assertEqual(d["total"], 3)
        self.assertEqual(d["groups"], [])
        self.assertTrue(all(h["group"] == "" for h in d["history"]))
        self.assertEqual(self._create("A", run_ids=["g-a1"]).get("touched"), 1)   # 坏了也还能重写

    def test_v2_partial_junk_dropped_item_by_item(self):
        """单条坏项丢掉其余照用：整体降级会让一次手改文件的笔误清空全部归组。"""
        self._write_map({"v": 2,
                         "groups": [{"id": "g-ok", "name": "甲", "note": None},
                                    {"id": "g-ok", "name": "重复 id"},
                                    "不是对象",
                                    {"id": "", "name": "无 id"},
                                    {"id": "g-noname", "name": "  "}],
                         "assign": {"g-a1": "g-ok", "g-a2": "g-gone", "": "g-ok",
                                    "g-b1": 5}})
        d = self._get()
        self.assertEqual(d["groups"], [{"id": "g-ok", "name": "甲", "note": "", "count": 1}])
        self.assertEqual(self._summary("g-a1")["group"], "甲")
        self.assertEqual(self._summary("g-a2")["group"], "")

    def test_groups_file_never_listed(self):
        """映射文件也在 results/ 且同样 .json 结尾，不能被当成一条历史记录。"""
        self._create("X", run_ids=["g-a1"])
        d = self._get()
        self.assertEqual(d["total"], 3)
        self.assertNotIn(MAP, json.dumps(d))

    def test_groups_file_unreachable_via_run_id_routes(self):
        """文件名以点开头 → run_id 白名单天然挡住，不需要额外开洞。"""
        self._create("X", run_ids=["g-a1"])
        for method in (self.client.get, self.client.delete):
            self.assertEqual(method("/api/bench/history/.history-groups").status_code, 400)

    def test_search_finds_by_group_name(self):
        self._create("量化对比", run_ids=["g-b1"])
        d = self.client.get("/api/bench/history?limit=1&q=量化对比").json()
        self.assertEqual([h["run_id"] for h in d["history"]], ["g-b1"])
        self.assertEqual(d["groups"][0]["count"], 1)   # 台账仍按全集数，不受 q 影响

    def test_group_can_be_written_before_archive_exists(self):
        """刚点「启动」时存档还没落盘（跑完才写），组名必须当场就能盖上。"""
        rid = "20990101-000000-abcd"
        r = self._create("夜间跑批", run_ids=[rid])
        self.assertEqual(r["touched"], 1)
        self.assertNotIn(rid, [h["run_id"] for h in self._get()["history"]])
        self.assertEqual(self._byname("夜间跑批")["count"], 0)   # 存档没在 → 不算一条
        _archive(rid)
        self.assertEqual(self._summary(rid)["group"], "夜间跑批")
        self.assertEqual(self._byname("夜间跑批")["count"], 1)

    def test_clear_removes_map_but_not_counted(self):
        self._create("A", run_ids=["g-a1"])
        r = self.client.delete("/api/bench/history")
        self.assertEqual(r.json()["removed"], 3)        # 只数存档，映射文件不充数
        self.assertFalse(os.path.exists(os.path.join(server.RESULTS_DIR, MAP)))

    # ---- 门禁：分组绝不写进存档 -----------------------------------------------------

    def test_group_ops_never_touch_archive_bytes(self):
        """防缝合怪的门禁断言：分组是视图层，删掉映射文件即全没、**存档一个字没改过**
        （ADR-0126 第 1 条 / ADR-0129 代价）。这条一改绿就意味着分组又缝回数据里了。"""
        before = _fingerprint()          # 每个 results/*.json 的 (字节 sha1, mtime_ns)
        a = self._create("A", note="甲", run_ids=["g-a1", "g-a2"])["created"][0]
        b = self._create("B", run_ids=["g-b1"])["created"][0]
        self._post([{"op": "rename", "id": a, "name": "A2", "note": "改过"}])
        self._post([{"op": "move", "id": b, "run_ids": ["g-a1"]}])
        self._post([{"op": "move", "id": "", "run_ids": ["g-a2"]}])
        self._post([{"op": "merge", "id": a, "into": b}])
        self._post([{"op": "dissolve", "id": b}])
        self.assertEqual(_fingerprint(), before, "动组改了存档字节：分组不再是视图层")
        for name in os.listdir(server.RESULTS_DIR):
            if name.endswith(".json") and name != MAP:
                with open(os.path.join(server.RESULTS_DIR, name), encoding="utf-8") as f:
                    self.assertNotIn("group", json.load(f).get("cfg", {}))


if __name__ == "__main__":
    unittest.main()
