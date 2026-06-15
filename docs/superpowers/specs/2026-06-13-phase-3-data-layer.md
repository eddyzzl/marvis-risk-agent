# Phase 3 — 数据层 + 数据文件处理包（函数级 spec，含内部伪代码）

## 文档状态

- 状态：待实施
- 日期：2026-06-13
- 上级蓝图：`2026-06-13-marvis-platform-blueprint.md`（第 7 节，数据层与 join 引擎）
- 前置依赖：Phase 1（Tool 契约/子进程 runner，data_ops 包以内置 Plugin 形式落地）
- 目标：交付数据基础设施 + 数据文件处理能力包。**核心是 join 引擎**：把人工拼表的隐性判断显性化为可机检诊断 + 强制确认。

## 捍卫的不变量

- **INV-3**：join 不静默执行。`propose_join_plan` 产诊断，`execute_join` 在 plan 里强制 `needs_confirmation`；执行后断言 `joined_rows <= anchor_rows`。
- **INV-1/INV-2**：数据层产出结构化（Dataset/ColumnProfile/JoinDiagnostics），不靠 LLM 判断拼接对错。
- **INV-5**：`sample_values` 脱敏；不持久化原始客户明细到记忆/审计。
- **INV-9**：路径 `as_posix()`；DuckDB/pandas 读写显式 encoding；大文件不进内存即崩。

## 模块布局

```text
marvis/data/
  __init__.py
  contracts.py      Dataset/ColumnProfile/ColumnFingerprint/JoinPlan/JoinSpec/KeyPair/JoinDiagnostics
  errors.py         数据层异常
  backend.py        DataBackend：DuckDB/pandas 混合后端
  excel_ingest.py   多 sheet + 合并表头拍平
  fingerprint.py    列值指纹（hash 家族 md5/sha256/... + 日期格式 + 大小写）
  schema_infer.py   列类型 + 语义角色 + target 检测
  sampler.py        采样
  profiler.py       列画像
  align.py          ColumnAligner：键字典 + 模糊兜底
  join_engine.py    JoinEngine：propose / diagnose / execute
  registry.py       DatasetRegistry
marvis/packs/data_ops/
  __init__.py
  manifest.json     7 个 tool 声明
  tools.py          tool_* 函数（包装 data/ 能力为 Tool 契约）
marvis/db.py   新增 datasets / joins 表 + DatasetRepository
```

新增依赖：`duckdb>=0.9`、`pyarrow>=12`（feather/parquet）、`openpyxl`（已有）、`rapidfuzz>=3`（模糊列名匹配）。

---

## Part A — 契约（`data/contracts.py`）

```python
@dataclass(frozen=True)
class ColumnFingerprint:
    value_kind: str          # raw_phone|raw_idcard|hash|date|numeric|categorical|unknown
    length_mode: int | None  # 最常见值长度
    regex_pattern: str | None
    is_hashed: bool          # 疑似哈希值（md5/sha1/sha256/...）
    hash_type: str | None    # md5|sha1|sha224|sha256|sha384|sha512（按 hex 长度判定）；非 hash 为 None
    hex_case: str | None     # lower|upper|mixed（hash 列大小写，用于统一规范化）；非 hash 为 None
    date_format: str | None  # 检测到的日期格式 strptime 模板（%Y%m%d / %Y-%m-%d / %Y/%m/%d ...）；非日期为 None

@dataclass(frozen=True)
class ColumnProfile:
    name: str
    dtype: str
    semantic_role: str       # id|phone|idcard|date|amount|target|score|categorical|numeric|unknown
    fingerprint: ColumnFingerprint
    null_rate: float
    cardinality: int
    sample_values: tuple     # 脱敏后前 N 个（INV-5）

@dataclass(frozen=True)
class Dataset:
    id: str
    task_id: str
    role: str                # sample|feature|derived|unknown
    source_path: str         # 相对 task_dir
    format: str              # csv|feather|parquet|xlsx
    sheet: str | None
    row_count: int
    columns: tuple[ColumnProfile, ...]
    has_target: bool
    target_col: str | None
    created_at: str

@dataclass(frozen=True)
class KeyPair:
    anchor_col: str
    feature_col: str
    match_method: str        # exact|exact_lower|date|hash:<algo>（如 hash:md5、hash:sha256）
    transform_side: str      # anchor|feature|both —— 对哪侧套规范化（raw 侧加 hash；两侧统一 lower/date）
    match_rate: float        # 用小样本数据实际试出来的命中率（核心：用数据验证能不能拼，不靠名字猜）
    resolved_by: str         # dictionary|fuzzy|empirical —— 这个键对怎么定的

@dataclass
class JoinDiagnostics:
    anchor_rows: int
    feature_rows: int
    feature_key_unique: bool
    matched_rows: int
    match_rate: float
    joined_rows_preview: int
    fan_out_detected: bool    # joined_rows_preview > anchor_rows
    shrink_detected: bool     # match_rate < SHRINK_WARN_THRESHOLD
    new_columns: int
    new_columns_null_rate: float

@dataclass
class JoinSpec:
    feature_dataset_id: str
    key_pairs: list[KeyPair]
    diagnostics: JoinDiagnostics
    dedup_strategy: str | None   # first|last|agg_mean|agg_max|abort|None
    confirmed: bool = False

@dataclass
class JoinPlan:
    id: str
    task_id: str
    anchor_dataset_id: str
    joins: list[JoinSpec]
    status: str              # draft|confirmed|executed|rejected
    result_dataset_id: str | None = None

# 常量
SHRINK_WARN_THRESHOLD = 0.5   # 命中率低于此告警
SMALL_SAMPLE_N = 5000         # 试拼/命中验证的小样本量
LARGE_ROW_THRESHOLD = 200_000 # 超过走 DuckDB
MIN_KEY_MATCH_RATE = 0.5      # 经数据验证命中率低于此，认为这对键拼不上
# hex 长度 → hash 算法（识别 md5/sha 家族；大小写不敏感）
HASH_HEX_LENGTHS = {32: "md5", 40: "sha1", 56: "sha224", 64: "sha256", 96: "sha384", 128: "sha512"}
# raw↔hash 拼接时，按命中率实际尝试的候选算法（顺序=优先级）。指纹已知 hash_type 时优先用它。
HASH_ALGO_CANDIDATES = ("md5", "sha256", "sha1", "sha512")
# 日期字段尝试解析的格式（统一转换用），覆盖字符型多种写法 + datetime
DATE_FORMATS = ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
                "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S")
```

- **测试要点**：dataclass 往返；常量可配置；`HASH_HEX_LENGTHS` 覆盖 md5/sha1/sha256/sha512。

---

## Part B — 后端抽象（`data/backend.py`，应对十万~千万行）

```python
def sql_string_literal(value: str) -> str:
    """DuckDB SQL 字符串字面量转义，用于文件路径等不可参数化位置。
    伪代码: return "'" + value.replace("'", "''") + "'"
    """

def sql_identifier(name: str, allowed_columns: set[str]) -> str:
    """DuckDB SQL 标识符转义，只允许来自已登记 schema 的列名。
    异常: DataSecurityError（列名不在 allowed_columns）。
    伪代码:
      if name not in allowed_columns: raise DataSecurityError(f"unknown column: {name}")
      return '"' + name.replace('"', '""') + '"'
    """

def parquet_rel(path: Path) -> str:
    """返回 read_parquet('...') 片段；路径必须先走 sql_string_literal。"""
    return f"read_parquet({sql_string_literal(path.as_posix())})"

def csv_rel(path: Path) -> str:
    """返回 read_csv_auto('...') 片段；路径必须先走 sql_string_literal。"""
    return f"read_csv_auto({sql_string_literal(path.as_posix())})"

class DataBackend:
    """统一 pandas / DuckDB 的数据操作。小数据 pandas，大数据 DuckDB（按行数阈值）。"""
    def __init__(self, datasets_root: Path):
        self._root = datasets_root

    def row_count(self, path: Path) -> int:
        """快速行数（不全载内存）。
        伪代码:
          if path.suffix == ".csv":
              return duckdb.sql(f"SELECT count(*) FROM {csv_rel(path)}").fetchone()[0]
          if path.suffix in (".parquet", ".feather"):
              return duckdb.sql(f"SELECT count(*) FROM {parquet_rel(path)}").fetchone()[0]
          # xlsx 已在 ingest 阶段转 parquet，这里不直接处理
        """

    def column_names(self, path: Path) -> list[str]:
        """只读表头/schema，不载数据。
        伪代码: duckdb DESCRIBE，或 pandas read_csv(nrows=0)/pyarrow schema。
        """

    def read_frame(self, path: Path, *, columns=None, nrows=None) -> pd.DataFrame:
        """读成 pandas DataFrame（建模/小数据用）。nrows 限制行数。
        不变量: INV-9 显式 encoding；大文件配 nrows 防 OOM。
        """

    def sample_rows(self, path: Path, n: int, *, seed: int) -> pd.DataFrame:
        """随机采样 n 行（大文件用 DuckDB USING SAMPLE，小文件 pandas.sample）。
        伪代码:
          total = self.row_count(path)
          if total <= n: return self.read_frame(path)
          if total > LARGE_ROW_THRESHOLD:
              q = f"SELECT * FROM {parquet_rel(path)} USING SAMPLE {int(n)} ROWS (system, {int(seed)})"
              return duckdb.sql(q).df()
          return self.read_frame(path).sample(n=n, random_state=seed)
        """

    def distinct_count(self, path: Path, columns: list[str]) -> int:
        """某些列组合的去重计数（判键唯一性用）。
        伪代码: duckdb SELECT count(*) FROM (SELECT DISTINCT cols FROM read...)。
        """

    def is_key_unique(self, path: Path, columns: list[str]) -> bool:
        """columns 组合是否唯一键。
        伪代码: distinct_count(path, columns) == row_count(path)。
        """

    def left_join(self, anchor_path, feature_path, key_pairs: list[KeyPair],
                  *, dedup_strategy: str | None, out_path: Path) -> int:
        """LEFT JOIN 锚定 anchor，写出结果到 out_path（parquet），返回结果行数。
        入参: key_pairs 已含 transform；dedup_strategy 处理 feature 键不唯一。
        出参: 结果行数。
        不变量: INV-3（左连接，结果行数应 <= anchor_rows；调用方断言）。
        伪代码:
          anchor_columns = set(self.column_names(anchor_path))
          feature_columns = set(self.column_names(feature_path))
          # 1. 构造 transform 后的 join 条件
          on_clauses = []
          for kp in key_pairs:
              a_col = "a." + sql_identifier(kp.anchor_col, anchor_columns)
              f_col = "b." + sql_identifier(kp.feature_col, feature_columns)
              a = _sql_transform(kp.transform, a_col)
              f = _sql_transform(kp.transform, f_col)
              on_clauses.append(f"{a} = {f}")
          on_sql = " AND ".join(on_clauses)
          # 2. dedup feature 侧（若需要）
          feature_rel = _dedup_sql(feature_path, key_pairs, dedup_strategy, allowed_columns=feature_columns)
          # 3. LEFT JOIN
          q = f"""COPY (
                    SELECT a.*, {_feature_cols_excluding_keys(...)}
                    FROM {parquet_rel(anchor_path)} a
                    LEFT JOIN ({feature_rel}) b ON {on_sql}
                  ) TO {sql_string_literal(out_path.as_posix())} (FORMAT parquet)"""
          duckdb.sql(q)
          return self.row_count(out_path)
        """

    def match_rate_for_method(self, anchor_path, anchor_keys, feature_path, feature_keys,
                              *, method: str, key_fingerprints, sample_n: int, seed: int) -> tuple[int, int]:
        """用指定 match_method 在小样本上实测命中率（"用数据实际试能不能拼"的执行点）。
        入参:
          anchor_keys/feature_keys: 对齐的键列名（同序）；
          method: exact|exact_lower|date|hash:<algo>；
          key_fingerprints: 每个键列的 ColumnFingerprint（决定哪侧是 raw、哪侧是 hash）。
        出参: (matched, sampled) —— match_rate = matched / sampled。
        不变量: 规范化对称——两侧最终落到同一可比形式后再比（hash 统一小写；date 统一 canonical）。
        伪代码:
          sample = self.sample_rows(anchor_path, sample_n, seed=seed)
          # 按 method + 指纹决定每侧 SQL 规范化表达；raw 侧套 hash、两侧统一 lower/canonical-date
          a_exprs = [_normalize_expr(col, method, side="anchor", fp=fp)
                     for col, fp in zip(anchor_keys, key_fingerprints)]
          f_exprs = [_normalize_expr(col, method, side="feature", fp=fp)
                     for col, fp in zip(feature_keys, key_fingerprints)]
          feat_keys_set = self._load_normalized_key_set(feature_path, f_exprs)
          matched = 0
          for row in sample:
              key = tuple(self._eval_normalized(row, col, method, fp) for col, fp in zip(anchor_keys, key_fingerprints))
              if key in feat_keys_set: matched += 1
          return matched, len(sample)
        """
```

辅助 `_normalize_expr(col, method, side, fp)` 把 (列, 方法, 侧, 指纹) 映射成 DuckDB SQL 规范化表达：

```text
method "exact"        → sql_identifier(col, allowed_columns)（原样，仅 strip）
method "exact_lower"  → lower(trim(sql_identifier(col, allowed_columns)))       # 统一小写（hash 大小写差异、字符键）
method "hash:md5"     → raw 侧: lower(md5(sql_identifier(...)))；hash 侧: lower(sql_identifier(...))
method "hash:sha256"  → raw 侧: lower(sha256(sql_identifier(...)))；hash 侧: lower(sql_identifier(...))
method "date"         → strftime(coalesce(try_strptime(sql_identifier(...), fmt1), ...), '%Y-%m-%d')
                         # 多格式尝试解析到 canonical 日期（yyyymmdd / yyyy-mm-dd / datetime 统一）
```

哪侧是 raw、哪侧是 hash 由 `fp.is_hashed` 判定（`side` + 指纹共同决定对 anchor 还是 feature 套 hash）。DuckDB 提供 `md5()`/`sha256()`；sha1/sha512 若内建缺失则用 UDF 兜底（实现时确认 DuckDB 版本的 hash 函数覆盖）。

- **测试要点**：小/大文件 row_count 一致；`is_key_unique` 正反例；`left_join` 行数正确、dedup 各策略生效；`match_rate_for_method` 对「明文 vs md5」「明文 vs sha256」「大写 hash vs 小写 hash」「yyyymmdd vs yyyy-mm-dd」各自经规范化后命中率正确；**SQL 注入安全**（路径走 `sql_string_literal`；列名必须来自白名单并双引号转义；包含单引号、双引号、空格、中文、SQL 关键字的路径/列名都有用例）。

> 安全注意：DuckDB SQL 拼接列名必须用白名单（列名来自已登记 `Dataset.columns`）或双引号转义，禁止把用户原始字符串拼进 SQL（参考 CODE_REVIEW P0-2 教训）。

---

## Part C — Excel 摄取（`data/excel_ingest.py`，多 sheet + 合并表头）

```python
@dataclass
class IngestReport:
    sheet: str
    header_rows: int          # 检测到的表头层数
    data_start_row: int
    flattened_columns: list[str]
    original_shape: tuple[int, int]

def list_sheets(path: Path) -> list[str]:
    """枚举 xlsx 所有 sheet 名。
    伪代码: openpyxl.load_workbook(path, read_only=True).sheetnames
    """

def detect_header_rows(raw: pd.DataFrame) -> int:
    """检测多行合并表头层数：从顶部找连续的"表头型"行（非数值占比高、有 NaN 合并痕迹），
       直到第一行"数据型"行。
    入参: raw（header=None 读入的前 N 行）。
    出参: 表头层数（≥1）。
    伪代码:
      for i in range(min(MAX_HEADER_ROWS, len(raw))):
          row = raw.iloc[i]
          if _looks_like_data_row(row):   # 数值/日期占比高 → 数据开始
              return max(i, 1)
      return 1
    """

def flatten_headers(raw: pd.DataFrame, header_rows: int) -> tuple[pd.DataFrame, list[str]]:
    """把多行合并表头拍平成单层 "父_子" 列名，返回 (数据DataFrame, 列名列表)。
    入参: raw（header=None）; header_rows。
    出参: (数据部分DataFrame, 拍平列名)。
    不变量: 合并单元格在 header 区表现为 NaN，需先按行 ffill 再纵向拼。
    伪代码:
      header_block = raw.iloc[:header_rows].copy()
      # 合并单元格：每一层横向 ffill（合并的右侧单元格是 NaN）
      for r in range(header_rows):
          header_block.iloc[r] = header_block.iloc[r].ffill()
      # 纵向拼接各层非空片段
      names = []
      for col in range(header_block.shape[1]):
          parts = [str(header_block.iloc[r, col]).strip()
                   for r in range(header_rows)
                   if pd.notna(header_block.iloc[r, col]) and str(header_block.iloc[r, col]).strip()]
          names.append("_".join(_dedupe_consecutive(parts)) or f"col_{col}")
      names = _disambiguate_duplicates(names)   # 同名列加后缀
      data = raw.iloc[header_rows:].reset_index(drop=True)
      data.columns = names
      return data, names
    """

def ingest_sheet(path: Path, sheet: str, out_dir: Path, *,
                 header_rows: int | None = None) -> tuple[Path, IngestReport]:
    """摄取单个 sheet：检测/拍平表头 → 写成 parquet（后续统一按 parquet 处理）。
    入参: path; sheet; out_dir 输出目录; header_rows 手动覆盖（None=自动检测）。
    出参: (parquet_path, IngestReport)。
    异常: DataIngestError（sheet 不存在/空表）。
    伪代码:
      raw = pd.read_excel(path, sheet_name=sheet, header=None, nrows=PREVIEW_ROWS)
      hr = header_rows or detect_header_rows(raw)
      full = pd.read_excel(path, sheet_name=sheet, header=None)
      data, names = flatten_headers(full, hr)
      out = out_dir / f"{_safe(sheet)}.parquet"
      data.to_parquet(out, index=False)
      return out, IngestReport(sheet, hr, hr, names, full.shape)
    """
```

- **测试要点**：单行表头正常；**两行合并表头**拍平成 `父_子`（造一个合并单元格 xlsx fixture）；合并单元格 ffill 正确；同名列消歧；空 sheet 抛错；多 sheet 各自摄取。

---

## Part D — 列值指纹（`data/fingerprint.py`，识别 hash 家族 / 日期格式 / 大小写）

```python
def fingerprint_column(series: pd.Series, *, sample_n: int = 1000, seed: int = 0) -> ColumnFingerprint:
    """对列采样算指纹，识别 raw_phone / raw_idcard / hash(md5/sha1/sha256/...) / date / numeric / categorical。
    入参: series; sample_n 采样量; seed。
    出参: ColumnFingerprint（含 hash_type、hex_case、date_format）。
    不变量: 指纹只识别"这列是什么"，不直接决定 transform；transform 由 align 阶段用数据实测决定。
    伪代码:
      s = series.dropna().astype(str).str.strip()
      if s.empty: return ColumnFingerprint("unknown", None, None, False, None, None, None)
      sample = s.sample(min(sample_n, len(s)), random_state=seed)
      length_mode = int(sample.str.len().mode().iloc[0])
      # 1) hash 家族：纯 hex 且长度命中 HASH_HEX_LENGTHS（大小写不敏感）
      if _frac_match(sample, r"^[0-9a-fA-F]+$") > 0.9 and length_mode in HASH_HEX_LENGTHS:
          algo = HASH_HEX_LENGTHS[length_mode]
          case = _detect_hex_case(sample)            # lower|upper|mixed
          return ColumnFingerprint("hash", length_mode, r"^[0-9a-fA-F]{%d}$" % length_mode,
                                   is_hashed=True, hash_type=algo, hex_case=case, date_format=None)
      # 2) 身份证（18 位，末位可 X）
      if _frac_match(sample, r"^\d{17}[\dXx]$") > 0.9:
          return ColumnFingerprint("raw_idcard", 18, r"^\d{17}[\dXx]$", True if False else False,
                                   None, None, None)  # is_hashed=False
      # 3) 手机号（11 位，1 开头）
      if _frac_match(sample, r"^1\d{10}$") > 0.9:
          return ColumnFingerprint("raw_phone", 11, r"^1\d{10}$", False, None, None, None)
      # 4) 日期：尝试 DATE_FORMATS，记录命中的格式
      fmt = _detect_date_format(sample)              # 返回命中的 strptime 模板或 None
      if fmt is not None:
          return ColumnFingerprint("date", None, None, False, None, None, date_format=fmt)
      # 5) 数值 / 类别
      if _frac_numeric(sample) > 0.9:
          return ColumnFingerprint("numeric", None, None, False, None, None, None)
      return ColumnFingerprint("categorical", length_mode, None, False, None, None, None)
    """

def candidate_match_methods(a: ColumnFingerprint, b: ColumnFingerprint) -> list[str]:
    """根据两列指纹列出"值得用数据实测的候选 match_method"。不判定能不能拼——那交给 align 用数据试。
    出参: 候选 method 列表（按优先级），空=语义不可能拼。
    不变量: raw↔hash 不靠猜单一算法，而是给出多个候选算法让 align 实测命中率。
    伪代码:
      # 同为明文同类 → 直接精确匹配（带大小写/strip 容错）
      if a.value_kind == b.value_kind and a.value_kind in ("raw_phone", "raw_idcard"):
          return ["exact", "exact_lower"]
      # 两边都是 hash → 统一小写后精确比（解决大小写差异）；长度不同则不可拼
      if a.value_kind == "hash" and b.value_kind == "hash":
          return ["exact_lower"] if a.length_mode == b.length_mode else []
      # 一边明文一边 hash → 对明文侧试多种 hash 算法（指纹已知 hash_type 时把它排最前）
      kinds = {a.value_kind, b.value_kind}
      if "hash" in kinds and ("raw_phone" in kinds or "raw_idcard" in kinds):
          known = (a.hash_type or b.hash_type)
          ordered = ([known] if known else []) + [x for x in HASH_ALGO_CANDIDATES if x != known]
          return [f"hash:{algo}" for algo in ordered]
      # 两边都是日期（可能不同格式）→ 统一 canonical 日期
      if a.value_kind == "date" and b.value_kind == "date":
          return ["date"]
      # 其它同类（categorical/numeric）→ 容错精确
      if a.value_kind == b.value_kind:
          return ["exact", "exact_lower"]
      return []
    """
```

辅助：
- `_frac_match(series, pattern) -> float`：匹配正则比例。
- `_detect_hex_case(sample) -> str`：全小写→"lower"，全大写→"upper"，混合→"mixed"。
- `_detect_date_format(sample) -> str | None`：依次试 `DATE_FORMATS`，返回首个解析成功率 >0.9 的模板；也处理 pandas datetime dtype（直接 "datetime"）。

- **测试要点**：
  - 32 位→hash(md5)、40 位→hash(sha1)、64 位→hash(sha256)、128 位→hash(sha512)。
  - 大写 hex hash → `hex_case="upper"`；小写 → "lower"。
  - `8 位 yyyymmdd` 字符串、`yyyy-mm-dd`、`yyyy/mm/dd`、datetime 列各识别为 date 且 `date_format` 正确。
  - `candidate_match_methods(raw_phone, hash(md5))` 返回 `["hash:md5","hash:sha256",...]`（已知 md5 排最前）。
  - 两个不同长度 hash → 返回 `[]`（拼不了）。
  - 两个不同格式日期 → 返回 `["date"]`。
  - **这组用例直接对应你说的：多种加密算法、大小写、日期多格式、用数据实测而非名字猜。**

---

## Part E — Schema 推断（`data/schema_infer.py`）

```python
def infer_column_profile(series: pd.Series, name: str, *, seed: int = 0) -> ColumnProfile:
    """单列画像：dtype + 指纹 + 语义角色 + null 率 + 基数 + 脱敏样例。
    伪代码:
      fp = fingerprint_column(series, seed=seed)
      role = detect_semantic_role(name, fp)
      return ColumnProfile(name=name, dtype=str(series.dtype), semantic_role=role,
          fingerprint=fp, null_rate=float(series.isna().mean()),
          cardinality=int(series.nunique()),
          sample_values=tuple(_desensitize(v, role) for v in series.dropna().head(N_SAMPLE)))
    """

def detect_semantic_role(name: str, fp: ColumnFingerprint) -> str:
    """综合列名 + 指纹判断语义角色。
    伪代码:
      lname = name.lower()
      if fp.value_kind in ("raw_phone",) or any(k in lname for k in PHONE_NAMES): return "phone"
      if fp.value_kind in ("raw_idcard",) or any(k in lname for k in ID_NAMES): return "idcard"
      if fp.value_kind == "hash":
          # hash 列（md5/sha256/...）靠列名归类到 phone/idcard/id
          if any(k in lname for k in PHONE_NAMES): return "phone"
          if any(k in lname for k in ID_NAMES): return "idcard"
          return "id"
      if fp.value_kind == "date" or any(k in lname for k in DATE_NAMES): return "date"
      if any(k in lname for k in TARGET_NAMES): return "target"
      if any(k in lname for k in SCORE_NAMES): return "score"
      if any(k in lname for k in AMOUNT_NAMES): return "amount"
      return "numeric" if fp.value_kind == "numeric" else "categorical"
    """

def infer_dataset_schema(df: pd.DataFrame, *, seed: int = 0) -> list[ColumnProfile]:
    """对所有列推断 profile。"""

def detect_target_column(profiles: list[ColumnProfile], df: pd.DataFrame) -> str | None:
    """检测 y 列：语义角色=target，或二值 {0,1} 且列名命中 TARGET_NAMES。
    伪代码:
      cands = [p.name for p in profiles if p.semantic_role == "target"]
      if cands: return cands[0]
      for p in profiles:
          vals = set(df[p.name].dropna().unique())
          if vals <= {0, 1, 0.0, 1.0} and any(k in p.name.lower() for k in TARGET_NAMES):
              return p.name
      return None
    """
```

字典常量（与 align 共享）：`PHONE_NAMES/ID_NAMES/DATE_NAMES/TARGET_NAMES/SCORE_NAMES/AMOUNT_NAMES`。

- **测试要点**：手机号/身份证/日期/目标列各正确归类；hash 列（md5/sha256）靠列名归到 phone/idcard；`detect_target_column` 识别 0/1 的 y 列；脱敏样例不泄露完整手机号/身份证（INV-5）。

---

## Part F — 采样与画像（`data/sampler.py` / `data/profiler.py`）

```python
# sampler.py
def sample_dataset(backend: DataBackend, path: Path, n: int, *,
                   strategy: str = "random", seed: int = 0,
                   stratify_col: str | None = None) -> pd.DataFrame:
    """采样。strategy: random|stratified|head。stratified 按 stratify_col 分层。
    伪代码:
      if strategy == "head": return backend.read_frame(path, nrows=n)
      if strategy == "stratified" and stratify_col:
          df = backend.read_frame(path)
          return df.groupby(stratify_col, group_keys=False).apply(
              lambda g: g.sample(min(len(g), max(1, n*len(g)//len(df))), random_state=seed))
      return backend.sample_rows(path, n, seed=seed)
    """

# profiler.py
def profile_dataset(backend: DataBackend, path: Path, *, seed: int = 0) -> list[ColumnProfile]:
    """对（采样后的）数据集算全列 profile（复用 schema_infer）。
    伪代码:
      sample = backend.sample_rows(path, PROFILE_SAMPLE_N, seed=seed)
      return infer_dataset_schema(sample, seed=seed)
    """
```

- **测试要点**：分层采样各层有代表；大文件走 DuckDB 采样不 OOM；profile 列数完整。

---

## Part G — 列对齐（`data/align.py`，语义匹配 + 数据实测）

> 核心哲学（用户原话）：**先按字段语义匹配（手机号拼手机号、身份证拼身份证），不同文件叫法不同要判断语义等价，然后用真实数据实际试能不能拼上。** 名字只用来缩小候选范围；最终是否配对、用什么 transform，由小样本实测命中率决定，不靠名字猜。

```python
KEY_DICTIONARY = {   # 仅用于"按语义把列归族"，不决定能不能拼
    "phone": ["phone", "mobile", "tel", "phone_no", "phone_md5", "mobile_md5", "tel_md5"],
    "idcard": ["idcard", "idnumber", "id_no", "cert_no", "id_md5", "idcard_md5"],
    "date": ["date", "applydate", "apply_date", "huisudate", "data_date", "dt", "create_date"],
}

class ColumnAligner:
    def __init__(self, backend: DataBackend):
        self._backend = backend     # 需要 backend 做"用数据实测"

    def align(self, anchor: Dataset, anchor_path: Path,
              feature: Dataset, feature_path: Path, *, seed: int = 0) -> list[KeyPair]:
        """找样本表↔特征表的 join 键对：语义归族缩候选 → 每个候选 method 用数据实测 → 取命中率达标者。
        入参: 两侧 Dataset + 物理路径; seed。
        出参: list[KeyPair]（每个含实测 match_rate + 最终 match_method；可能为空=没拼上的键）。
        不变量: 名字只缩候选，最终靠 `MIN_KEY_MATCH_RATE` 的实测命中率定夺（用数据试）。
        """
        # 伪代码:
        pairs = []
        # 1) 语义归族：按字典名 + semantic_role，把两侧列分到 phone/idcard/date 等族
        for family in ("phone", "idcard", "date"):
            a_cols = self._family_columns(anchor.columns, family)
            f_cols = self._family_columns(feature.columns, family)
            for a in a_cols:
                best = self._resolve_by_data(a, f_cols, anchor_path, feature_path, seed)
                if best is not None:           # 实测命中率达标才收
                    pairs.append(best)
        # 2) 模糊兜底：字典没覆盖到、但列名相似的列对，同样走数据实测
        if not pairs:
            pairs += self._fuzzy_resolve(anchor, anchor_path, feature, feature_path, seed)
        return _dedupe_keypairs(pairs)

    def _resolve_by_data(self, anchor_col: ColumnProfile, feature_cols: list[ColumnProfile],
                         anchor_path, feature_path, seed) -> KeyPair | None:
        """对一个 anchor 列，在候选 feature 列 × 候选 method 里，用小样本实测命中率，取最高且达标者。
        出参: 命中率最高且 >= MIN_KEY_MATCH_RATE 的 KeyPair；都不达标返回 None。
        不变量: 这是"用数据实际尝试能不能拼"的核心——raw↔hash 试 md5/sha256/sha1，日期试 canonical 化。
        伪代码:
          best = None
          for f in feature_cols:
              methods = candidate_match_methods(anchor_col.fingerprint, f.fingerprint)  # Part D
              for method in methods:
                  matched, sampled = self._backend.match_rate_for_method(
                      anchor_path, [anchor_col.name], feature_path, [f.name],
                      method=method, key_fingerprints=[_pair_fp(anchor_col, f)],
                      sample_n=SMALL_SAMPLE_N, seed=seed)
                  rate = matched / sampled if sampled else 0.0
                  if rate >= MIN_KEY_MATCH_RATE and (best is None or rate > best.match_rate):
                      best = KeyPair(anchor_col=anchor_col.name, feature_col=f.name,
                                     match_method=method,
                                     transform_side=_raw_side(anchor_col, f, method),
                                     match_rate=round(rate, 4), resolved_by="empirical")
          return best
        """

    def _fuzzy_resolve(self, anchor, anchor_path, feature, feature_path, seed) -> list[KeyPair]:
        """字典没覆盖时：rapidfuzz 列名相似度筛候选，再走同样的数据实测。
        伪代码:
          from rapidfuzz import fuzz
          out = []
          for a in anchor.columns:
              cands = [f for f in feature.columns if fuzz.ratio(a.name.lower(), f.name.lower()) >= FUZZY_NAME_THRESHOLD]
              kp = self._resolve_by_data(a, cands, anchor_path, feature_path, seed)
              if kp: out.append(replace(kp, resolved_by="fuzzy"))
          return out
        """

    def _family_columns(self, columns, family) -> list[ColumnProfile]:
        """按字典名 + semantic_role 把列归到某族（phone/idcard/date）。"""
```

辅助：
- `_raw_side(anchor_col, feature_col, method)`：method 是 `hash:*` 时，返回非 hashed 的那一侧（"anchor"/"feature"）作为 `transform_side`；`date`/`exact_lower` 返回 "both"；`exact` 返回 "both"（仅 strip）。
- `_pair_fp(a, f)`：给 backend 传这对列的指纹，让它判断哪侧套 hash。

- **测试要点**：
  - phone 族 `mobile`(明文) ↔ `phone_md5`(md5)：归族→候选 `hash:md5/sha256/...`→实测 md5 命中高→`KeyPair(match_method="hash:md5", transform_side="anchor", match_rate≈1.0)`。
  - phone 族 明文 ↔ **sha256**：实测 sha256 命中高，md5 命中≈0 → 选 `hash:sha256`（验证多算法实测）。
  - 大写 hash ↔ 小写 hash：`exact_lower` 命中（大小写统一）。
  - date 族 `applydate`(yyyymmdd) ↔ `huisudate`(yyyy-mm-dd)：`date` method canonical 化后命中。
  - 名字像但数据对不上（match_rate < 阈值）→ **不配对**（这正是"用数据否决名字误配"）。
  - 字典外列走模糊兜底 + 数据实测；没拼上的键正常返回空。

---

## Part H — Join 引擎（`data/join_engine.py`，平台最高风险，核心）

```python
class JoinEngine:
    def __init__(self, backend: DataBackend, aligner: ColumnAligner,
                 registry: "DatasetRegistry", repo: "DatasetRepository"):
        self._backend = backend
        self._aligner = aligner
        self._registry = registry
        self._repo = repo

    def propose_join_plan(self, anchor_id: str, feature_ids: list[str],
                          task_id: str, *, seed: int = 0) -> JoinPlan:
        """生成 JoinPlan：对每个特征表 align 键 + 小样本命中验证 + 试拼诊断。不执行。
        入参: anchor_id 样本表; feature_ids 特征表; task_id; seed。
        出参: JoinPlan(status="draft")，每个 JoinSpec 带完整诊断和待选 dedup_strategy。
        异常: DatasetNotFoundError。
        不变量: INV-3（只产计划+诊断，不执行）。
        """
        # 伪代码:
        anchor = self._registry.get(anchor_id)
        anchor_path = self._registry.resolve_path(anchor_id)
        specs = []
        for fid in feature_ids:
            feat = self._registry.get(fid)
            feat_path = self._registry.resolve_path(fid)
            # align 已"用数据实测"出 key_pairs（含 match_method + match_rate）
            key_pairs = self._aligner.align(anchor, anchor_path, feat, feat_path, seed=seed)
            diag = self.diagnose_join(anchor, anchor_path, feat, feat_path, key_pairs, seed=seed)
            # 默认 dedup 策略：键唯一→None；不唯一→待用户选（先置 None，前端强制选）
            specs.append(JoinSpec(feature_dataset_id=fid, key_pairs=key_pairs,
                                  diagnostics=diag, dedup_strategy=None, confirmed=False))
        plan = JoinPlan(id=_new_id(), task_id=task_id, anchor_dataset_id=anchor_id,
                        joins=specs, status="draft")
        self._repo.create_join_plan(plan)
        return plan

    def diagnose_join(self, anchor, anchor_path, feature, feature_path,
                      key_pairs: list[KeyPair], *, seed: int) -> JoinDiagnostics:
        """对单个特征表试拼诊断（小样本外推 + 键唯一性 + fan-out/shrink 检测）。
        入参: anchor/feature Dataset + 路径; key_pairs; seed。
        出参: JoinDiagnostics。
        不变量: INV-3 的诊断来源——fan_out（拼后膨胀）和 shrink（命中过低）必须标出。
        """
        # 伪代码:
        anchor_rows = anchor.row_count
        feature_rows = feature.row_count
        if not key_pairs:
            return JoinDiagnostics(anchor_rows, feature_rows, feature_key_unique=False,
                matched_rows=0, match_rate=0.0, joined_rows_preview=0,
                fan_out_detected=False, shrink_detected=True,   # 没键=无法拼
                new_columns=0, new_columns_null_rate=1.0)
        feature_keys = [kp.feature_col for kp in key_pairs]
        anchor_keys = [kp.anchor_col for kp in key_pairs]
        method = key_pairs[0].match_method   # 复合键各列同 method 族；混合时取最严格
        # 1) 特征表键唯一性（fan-out 源）—— 用规范化后的键判唯一性（hash/date 规范化后才准）
        key_unique = self._backend.is_key_unique(feature_path, feature_keys)
        # 2) 命中率：复用 align 已实测的 match_rate（多列复合键取联合命中率）
        if len(key_pairs) == 1:
            match_rate = key_pairs[0].match_rate
            matched = int(match_rate * SMALL_SAMPLE_N)
        else:
            # 复合键需联合实测（单列命中率不能简单相乘）
            matched, sampled = self._backend.match_rate_for_method(
                anchor_path, anchor_keys, feature_path, feature_keys,
                method=method, key_fingerprints=_key_fps(anchor, feature, key_pairs),
                sample_n=SMALL_SAMPLE_N, seed=seed)
            match_rate = matched / sampled if sampled else 0.0
        # 3) 试拼行数外推：键唯一→拼后≈anchor_rows；不唯一→可能膨胀
        if key_unique:
            joined_preview = anchor_rows
            fan_out = False
        else:
            # 估算平均重复倍数
            dup_factor = feature_rows / max(1, self._backend.distinct_count(feature_path, feature_keys))
            joined_preview = int(anchor_rows * match_rate * dup_factor + anchor_rows * (1 - match_rate))
            fan_out = joined_preview > anchor_rows
        shrink = match_rate < SHRINK_WARN_THRESHOLD
        new_cols = len([c for c in feature.columns if c.name not in {a.name for a in anchor.columns}])
        return JoinDiagnostics(anchor_rows, feature_rows, key_unique, matched,
            round(match_rate, 4), joined_preview, fan_out, shrink, new_cols,
            new_columns_null_rate=round(1 - match_rate, 4))

    def execute_join_plan(self, join_plan_id: str, *, out_dir: Path) -> Dataset:
        """执行已确认的 JoinPlan：逐表 LEFT JOIN 锚定样本表，断言行数不变量。
        入参: join_plan_id; out_dir 结果输出目录。
        出参: 结果 Dataset（role="derived"）。
        异常:
          JoinNotConfirmedError: 有未确认 JoinSpec；
          FanOutError: 执行后 joined_rows > anchor_rows（INV-3 硬断言）；
          DedupRequiredError: 键不唯一但未选 dedup_strategy。
        不变量: INV-3——左连接锚定，结果行数必须 <= anchor_rows，否则 abort + 报错。
        """
        # 伪代码:
        plan = self._repo.load_join_plan(join_plan_id)
        if any(not js.confirmed for js in plan.joins):
            raise JoinNotConfirmedError("all joins must be confirmed before execute (INV-3)")
        anchor_path = self._registry.resolve_path(plan.anchor_dataset_id)
        anchor_rows = self._registry.get(plan.anchor_dataset_id).row_count
        current_path = anchor_path
        for js in plan.joins:
            if not js.diagnostics.feature_key_unique and js.dedup_strategy in (None, "abort"):
                raise DedupRequiredError(f"feature {js.feature_dataset_id} key not unique; choose dedup strategy")
            feat_path = self._registry.resolve_path(js.feature_dataset_id)
            out_path = out_dir / f"join_{_new_id()}.parquet"
            joined_rows = self._backend.left_join(current_path, feat_path, js.key_pairs,
                                                  dedup_strategy=js.dedup_strategy, out_path=out_path)
            # INV-3 硬断言
            if joined_rows > anchor_rows:
                raise FanOutError(f"join produced {joined_rows} > anchor {anchor_rows} rows (fan-out)")
            current_path = out_path
        # 登记结果 Dataset
        result = self._registry.register_existing(current_path, task_id=plan.task_id,
                                                  role="derived", anchor_target=plan.anchor_dataset_id)
        self._repo.set_join_plan_executed(join_plan_id, result.id)
        return result

    def confirm_join_spec(self, join_plan_id: str, feature_dataset_id: str,
                          *, dedup_strategy: str | None) -> None:
        """用户确认单个 JoinSpec（设 dedup_strategy + confirmed=True）。
        异常: DedupRequiredError（键不唯一却选了 None）。
        伪代码:
          plan = self._repo.load_join_plan(join_plan_id)
          js = _find_spec(plan, feature_dataset_id)
          if not js.diagnostics.feature_key_unique and dedup_strategy in (None, "abort"):
              raise DedupRequiredError(...)
          js.dedup_strategy = dedup_strategy; js.confirmed = True
          self._repo.update_join_spec(join_plan_id, js)
        """
```

- **测试要点**（最重要，对应你说的全部坑）：
  - **键唯一表**：propose→diagnose 显示 `fan_out_detected=False`、`joined_rows_preview≈anchor_rows`；execute 后行数=anchor_rows。
  - **键不唯一表（会笛卡尔膨胀）**：`fan_out_detected=True`、`feature_key_unique=False`；未选 dedup 执行抛 `DedupRequiredError`；选 first/last 后行数=anchor_rows。
  - **命中率低（拼完样本反而少）**：`shrink_detected=True`、`match_rate` 低、`new_columns_null_rate` 高。
  - **raw vs 多种 hash**：明文手机号 anchor + md5 feature → `match_method="hash:md5"` 命中正常；换 sha256 feature → 自动实测选 `hash:sha256`（验证多算法实测闭环）。
  - **大小写差异 hash**：大写 hash ↔ 小写 hash → `exact_lower` 统一小写后命中。
  - **日期多格式**：`yyyymmdd` ↔ `yyyy-mm-dd` ↔ datetime → `date` method canonical 化后命中。
  - **名字像但数据对不上**：列名相似但实测命中率 < `MIN_KEY_MATCH_RATE` → 不配对（用数据否决名字误配）。
  - **执行后膨胀**：构造一个绕过 dedup 的 fan-out，断言 `FanOutError` 触发（INV-3 最后防线）。
  - 未确认就 execute → `JoinNotConfirmedError`。
  - 多特征表链式 join + 复合键（手机号+身份证+日期一起拼），每步断言行数不变量。

---

## Part I — 数据集登记（`data/registry.py`）

```python
class DatasetRegistry:
    def __init__(self, repo: "DatasetRepository", backend: DataBackend, datasets_root: Path):
        self._repo = repo; self._backend = backend; self._root = datasets_root

    def register_from_upload(self, task_id: str, source_path: Path, *,
                             role: str = "unknown", seed: int = 0) -> Dataset:
        """登记一个新上传文件：转 parquet（若需）→ profile → 落库。
        入参: task_id; source_path; role; seed。
        出参: Dataset。
        伪代码:
          fmt = _detect_format(source_path)
          parquet_path = self._normalize_to_parquet(source_path, fmt)   # csv/feather→parquet
          profiles = profile_dataset(self._backend, parquet_path, seed=seed)
          target = detect_target_column(profiles, self._backend.sample_rows(parquet_path, 1000, seed=seed))
          ds = Dataset(id=_new_id(), task_id=task_id, role=role,
                       source_path=str(parquet_path.relative_to(self._root)), format="parquet",
                       sheet=None, row_count=self._backend.row_count(parquet_path),
                       columns=tuple(profiles), has_target=target is not None,
                       target_col=target, created_at=_now_iso())
          self._repo.create_dataset(ds)
          return ds
        """

    def register_existing(self, parquet_path: Path, *, task_id: str, role: str,
                          anchor_target: str | None = None, seed: int = 0) -> Dataset:
        """登记已生成的 parquet（如 join 结果）。target 从 anchor 继承。"""

    def get(self, dataset_id: str) -> Dataset:
        """取 Dataset。异常: DatasetNotFoundError。"""

    def list_for_task(self, task_id: str) -> list[Dataset]:
        """列任务下所有数据集。"""

    def resolve_path(self, dataset_id: str) -> Path:
        """dataset_id → 物理 parquet 路径（绝对）。
        伪代码: return self._root / self.get(dataset_id).source_path
        """

    def set_role(self, dataset_id: str, role: str) -> None:
        """改 role（如用户指定哪个是样本表）。"""
```

- **测试要点**：csv/feather 登记转 parquet；profile + target 检测正确；role 设置；resolve_path 正确；list_for_task 隔离。

---

## Part J — 持久层（`db.py` 新增 + `DatasetRepository`）

```sql
CREATE TABLE IF NOT EXISTS datasets (
  id TEXT PRIMARY KEY, task_id TEXT NOT NULL, role TEXT NOT NULL,
  source_path TEXT NOT NULL, format TEXT NOT NULL, sheet TEXT,
  row_count INTEGER NOT NULL, columns_json TEXT NOT NULL,
  has_target INTEGER NOT NULL, target_col TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS joins (
  id TEXT PRIMARY KEY, task_id TEXT NOT NULL, anchor_dataset_id TEXT NOT NULL,
  joins_json TEXT NOT NULL, status TEXT NOT NULL, result_dataset_id TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_datasets_task ON datasets(task_id);
```

```python
class DatasetRepository:
    def create_dataset(self, ds: Dataset) -> None: ...        # columns_json = json of ColumnProfile list
    def get_dataset(self, dataset_id) -> dict | None: ...
    def list_datasets(self, task_id) -> list[dict]: ...
    def set_dataset_role(self, dataset_id, role) -> None: ...
    def create_join_plan(self, plan: JoinPlan) -> None: ...
    def load_join_plan(self, plan_id) -> JoinPlan: ...
    def update_join_spec(self, plan_id, spec: JoinSpec) -> None: ...   # 重写 joins_json 中该 spec
    def set_join_plan_executed(self, plan_id, result_dataset_id) -> None: ...
```

- **测试要点**：dataset 往返（含 columns_json 反序列化回 ColumnProfile）；join plan 往返；update_join_spec 只改目标 spec；状态流转。

---

## Part J-2 — HTTP API 契约（供 Frontend V2 调用）

> 本节只定义数据集/拼表的 HTTP 面。确定性计算仍在 `data/` 与 `packs/data_ops/`，API 层只做请求校验、权限边界、状态码和 payload 转换。

### J2-1 数据集端点

```text
GET /api/tasks/{task_id}/datasets
→ 200 {
  "datasets": [{
    "id": "ds_...",
    "task_id": "...",
    "role": "sample|feature|joined|unknown",
    "source_name": "features.xlsx",
    "format": "parquet",
    "sheet": "Sheet1",
    "row_count": 12345,
    "columns": [{ "name": "mobile_md5", "semantic_role": "phone", "dtype": "string", "is_hashed": true }],
    "has_target": true,
    "target_col": "y"
  }]
}
```

```text
POST /api/tasks/{task_id}/datasets/upload
Content-Type: multipart/form-data
fields:
  file: csv|xlsx|parquet|feather
  role?: sample|feature|unknown
  sheet?: string              # xlsx 指定单 sheet；缺省则摄取全部 sheet
→ 201 {
  "datasets": [{...}],
  "reports": [{ "sheet": "Sheet1", "warnings": [] }]
}
```

```text
GET /api/datasets/{dataset_id}/preview?rows=50
→ 200 { "columns": ["a", "b"], "rows": [{"a": 1, "b": "x"}], "truncated": true }
```

错误码：`404` 任务/数据集不存在；`422` 文件类型、sheet、role 或 query 参数非法；`400` 摄取失败但不是参数错误；上传接口必须复用 Phase 0 `apiPost(FormData)` 的 multipart 契约。

### J2-2 Join 端点

```text
POST /api/tasks/{task_id}/joins/propose
body: {
  "anchor_dataset_id": "ds_sample",
  "feature_dataset_ids": ["ds_feature_1", "ds_feature_2"]
}
→ 201 {
  "join_plan_id": "join_...",
  "status": "draft",
  "joins": [{
    "feature_id": "ds_feature_1",
    "key_pairs": [{
      "anchor_col": "mobile",
      "feature_col": "mobile_md5",
      "match_method": "hash:md5",
      "match_rate": 0.982,
      "resolved_by": "data_test"
    }],
    "diagnostics": { "anchor_rows": 1000, "matched_rows": 982, "fan_out_risk": false },
    "confirmed": false
  }]
}
```

```text
GET /api/joins/{join_plan_id}
→ 200 { "join_plan_id": "...", "status": "draft|confirmed|executed", "joins": [...] }

POST /api/joins/{join_plan_id}/confirm
body: {
  "feature_id": "ds_feature_1",
  "confirmed": true,
  "key_pairs": [...],              # 用户可修正；为空则沿用 propose 结果
  "dedup_strategy": "first|last|max_date|null"
}
→ 200 { "join_plan_id": "...", "joins": [...] }

POST /api/joins/{join_plan_id}/execute
→ 200 {
  "result_dataset_id": "ds_joined",
  "anchor_rows": 1000,
  "joined_rows": 1000,
  "fan_out": false,
  "warnings": []
}
```

错误码：`404` join plan 或 dataset 不存在；`422` key_pairs/feature_id/dedup_strategy 非法；`409` 仍有未确认 join、fan-out 风险未处理、行数不变量失败或 plan 已执行不可重复执行。

### J2-3 API 测试要点

- `list datasets` 按 `task_id` 隔离，不泄露其他任务数据集。
- `upload dataset` 支持 multipart；xlsx 多 sheet 返回多个 dataset；非法 sheet 返回 422。
- `preview` 有 rows 上限，默认 50，最大值由 settings 控制。
- `propose join` 不执行 join，不产生 result dataset。
- `confirm join` 可保存用户修正键对；非法列名被 schema 白名单拒绝。
- `execute join` 必须所有 join confirmed；fan-out/行数异常返回 409，不静默写出成功结果。

---

## Part K — data_ops 能力包（`packs/data_ops/`）

`manifest.json` 声明 7 个 tool（input/output schema 略写要点）：

```python
# tools.py —— 每个 tool 包装 data/ 能力为 Tool 契约，determinism=deterministic

def tool_ingest_excel(inputs: dict, ctx) -> dict:
    """摄取 xlsx 全部或指定 sheet。
    inputs: {path: str, sheets?: [str]}。
    output: {datasets: [{id, sheet, row_count, columns}], reports: [...]}。
    伪代码: for sheet in (inputs.sheets or list_sheets(path)):
              parquet, report = ingest_sheet(...); ds = registry.register_existing(parquet, role="feature")
    """

def tool_infer_schema(inputs: dict, ctx) -> dict:
    """对已登记 dataset 推断/刷新 schema。
    inputs: {dataset_id}。 output: {columns: [...], has_target, target_col}。
    """

def tool_align_columns(inputs: dict, ctx) -> dict:
    """样本表 vs 特征表们的候选键对（语义归族 + 数据实测）。
    inputs: {anchor_id, feature_ids: [str]}。
    output: {alignments: [{feature_id, key_pairs:
             [{anchor_col, feature_col, match_method, transform_side, match_rate, resolved_by}]}]}。
    不变量: match_rate 是用数据实测的命中率；前端据此让用户判断键对是否可信。
    """

def tool_propose_join(inputs: dict, ctx) -> dict:
    """生成 JoinPlan + 诊断（不执行）。
    inputs: {anchor_id, feature_ids}。
    output: {join_plan_id, joins: [{feature_id, key_pairs, diagnostics}]}。
    不变量: INV-3——只产计划。
    """

def tool_execute_join(inputs: dict, ctx) -> dict:
    """执行已确认 JoinPlan。
    inputs: {join_plan_id}。
    output: {result_dataset_id, anchor_rows, joined_rows, fan_out: false}。
    异常路径: 未确认/未选 dedup/fan-out → tool 返回 error（runner 收进 ToolResult.ok=False）。
    不变量: INV-3——在 Plan 里这个 tool 的 step 必须 needs_confirmation=True（PlanValidator 强制）。
    """

def tool_clean_format(inputs: dict, ctx) -> dict:
    """格式清洗：去空白/统一大小写/类型转换/去重列。
    inputs: {dataset_id, ops: [{col, op}]}。 output: {dataset_id, changed_columns}。
    """

def tool_dedup_rows(inputs: dict, ctx) -> dict:
    """按键去重。
    inputs: {dataset_id, keys: [str], strategy: "first"|"last"}。
    output: {dataset_id, removed_rows}。
    """
```

`manifest.json` 中 `tool_execute_join` 的 ToolSpec 标 `side_effects=["write:dataset"]`，并在文档注明：编排层把它编进 Plan 时必须设 `needs_confirmation=True`（由 `PlanValidator._check_join_gates` 在 Phase 2 强制）。

- **测试要点**：每个 tool 经 `ToolRunner.invoke` 子进程往返；`tool_propose_join` 输出诊断完整；`tool_execute_join` 未确认时 `ok=False`；schema 校验生效。

---

## Part L — 测试计划汇总

| 文件 | 覆盖 |
|------|------|
| `tests/test_data_contracts.py` | dataclass 往返、常量 |
| `tests/test_data_backend.py` | row_count/sample/is_key_unique/left_join/match_rate、**SQL 列名安全** |
| `tests/test_data_excel_ingest.py` | **多 sheet、合并表头拍平**、同名消歧、空表 |
| `tests/test_data_fingerprint.py` | **hash 家族(md5/sha1/sha256/sha512)识别、大小写、日期多格式、candidate_match_methods** |
| `tests/test_data_schema_infer.py` | 语义角色、hash 列靠名归类、target 检测、脱敏 |
| `tests/test_data_align.py` | 语义归族、**数据实测选算法(md5/sha256)**、大小写统一、日期 canonical、名字像但数据否决 |
| `tests/test_data_join_engine.py` | **键唯一/不唯一/低命中/raw-多hash/大小写/日期多格式/fan-out 断言/未确认/复合键链式**（核心） |
| `tests/test_data_registry.py` | 登记转 parquet、profile、resolve_path |
| `tests/test_data_db.py` | DatasetRepository 往返、join plan 流转 |
| `tests/test_data_api.py` | datasets/preview/upload/joins HTTP 契约、FormData、422/404/409 |
| `tests/test_data_ops_pack.py` | 7 个 tool 经 runner 往返 |

数据 fixture：造小规模带 y 的样本表 + 多个特征表，覆盖你描述的全部真实场景——一个键唯一表、一个键不唯一会膨胀的表、一个 **md5** 手机号键表、一个 **sha256** 键表、一个**大写 hash** 表、一个**日期 yyyymmdd vs yyyy-mm-dd** 表、一个合并表头 xlsx、一个**复合键（手机号+身份证+日期）**表。

---

## Part M — 任务执行顺序

```text
1. A 契约              （无依赖）
2. J DB + DatasetRepository（依赖 A + Phase 0 connect）
3. B backend           （依赖 A，加 duckdb/pyarrow 依赖）
4. D fingerprint       （依赖 A）
5. E schema_infer      （依赖 A,D）
6. F sampler/profiler   （依赖 B,E）
7. C excel_ingest      （依赖 A）
8. I registry          （依赖 B,E,F,J）
9. G align             （依赖 A,D + rapidfuzz）
10. H join_engine       （依赖 B,G,I,J；核心，最花时间）
11. K data_ops pack     （依赖全部 + Phase 1 Tool 契约）
12. J2 HTTP API         （依赖 I,J,K；供 Frontend V2 调用）
13. L 测试 + 回归
```

每项 atomic commit。Phase 3 完成标志：能摄取多 sheet 合并表头 xlsx，自动识别样本表/特征表列语义，对真实场景（含 md5 键、键不唯一表）产出带 fan-out/shrink 告警的 JoinPlan，逐表确认后 LEFT JOIN 锚定执行且行数不变量被强制；7 个 data_ops tool 经子进程 runner 可用。

---

*Phase 3 把"用户上传一堆乱七八糟的表"变成"平台理解的结构化数据集 + 安全可控的拼接"。join 引擎是整个平台落地价值最直接、风险也最高的一块——它替代的正是风控建模里最耗人、最易错的环节。*
