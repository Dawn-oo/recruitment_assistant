import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


# ============================================================
# 文件路径
# ============================================================
project_root = Path(__file__).resolve().parents[1]
input_path = project_root / "jobs_raw.json"

INPUT_FILE = Path(input_path)

REPORT_FILE = Path("data/validated/quality_report.json")


# ============================================================
# 质量问题定义
# ============================================================

@dataclass
class QualityIssue:
    field: str
    issue_type: str
    level: Literal["warning", "error"]
    message: str


# ============================================================
# 辅助函数
# ============================================================

def normalize_for_compare(text: str) -> str:
    """
    只用于重复值比较。
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    return text


# ============================================================
# 字符串字段质量校验
# ============================================================
def validate_text_quality(
    value: str | None,
    field_path: str,
    required: bool = True,
) -> list[QualityIssue]:

    issues = []

    # None 单独判断
    if value is None:
        if required:
            issues.append(
                QualityIssue(
                    field=field_path,
                    issue_type="null_value",
                    level="error",
                    message="必填字段为 None",
                )
            )

        return issues

    # 空字符串 / 纯空白
    if not value.strip():
        if required:
            issues.append(
                QualityIssue(
                    field=field_path,
                    issue_type="empty_value",
                    level="error",
                    message="必填字符串为空或只包含空白字符",
                )
            )

        return issues

    # 首尾空白
    if value != value.strip():
        issues.append(
            QualityIssue(
                field=field_path,
                issue_type="leading_trailing_whitespace",
                level="warning",
                message="字符串存在首尾空白字符",
            )
        )

    # 换行或 Tab
    if "\n" in value or "\r" in value or "\t" in value:
        issues.append(
            QualityIssue(
                field=field_path,
                issue_type="control_whitespace",
                level="warning",
                message="字符串中存在换行符或 Tab",
            )
        )

    # 连续多个空格
    if re.search(r" {2,}", value):
        issues.append(
            QualityIssue(
                field=field_path,
                issue_type="repeated_whitespace",
                level="warning",
                message="字符串中存在连续多个空格",
            )
        )

    # 中文标点前多余空白
    if re.search(r"\s+[，。；：、！？]", value):
        issues.append(
            QualityIssue(
                field=field_path,
                issue_type="punctuation_whitespace",
                level="warning",
                message="中文标点前存在多余空白",
            )
        )

    return issues

# ============================================================
# list[str] 字段质量校验
# ============================================================

def validate_string_list_quality(
    values: list[str],
    field_path: str,
    empty_level: Literal["warning", "error"] = "warning",
) -> list[QualityIssue]:

    issues = []

    # --------------------------------------------------------
    # 1. 整个列表为空
    # --------------------------------------------------------

    if not values:

        issues.append(
            QualityIssue(
                field=field_path,
                issue_type="empty_list",
                level=empty_level,
                message="列表为空",
            )
        )

        return issues

    # 用于完全重复检查
    exact_seen = {}

    # 用于空白格式规范化后的重复检查
    normalized_seen = {}

    for index, value in enumerate(values):

        current_path = f"{field_path}[{index}]"

        # ----------------------------------------------------
        # 2. 每个字符串本身的质量
        # ----------------------------------------------------

        issues.extend(
            validate_text_quality(
                value=value,
                field_path=current_path,
                required=True,
            )
        )

        # 空项已经在上面记录
        if not value.strip():
            continue

        # ----------------------------------------------------
        # 3. 完全重复
        # ----------------------------------------------------

        if value in exact_seen:

            issues.append(
                QualityIssue(
                    field=current_path,
                    issue_type="duplicate_item",
                    level="warning",
                    message=(
                        f"与 {field_path}"
                        f"[{exact_seen[value]}] 完全重复"
                    ),
                )
            )

        else:
            exact_seen[value] = index

        # ----------------------------------------------------
        # 4. 仅空白格式不同导致的重复
        # ----------------------------------------------------

        normalized = normalize_for_compare(value)

        if normalized in normalized_seen:

            previous_index = normalized_seen[
                normalized
            ]

            previous_value = values[
                previous_index
            ]

            # 如果已经是完全相同，
            # 就不重复报告 normalized_duplicate
            if value != previous_value:

                issues.append(
                    QualityIssue(
                        field=current_path,
                        issue_type="normalized_duplicate",
                        level="warning",
                        message=(
                            f"与 {field_path}"
                            f"[{previous_index}] "
                            f"在忽略空白格式后重复"
                        ),
                    )
                )

        else:
            normalized_seen[normalized] = index

    return issues


# ============================================================
# responsibilities 质量校验
# ============================================================

def validate_responsibilities_quality(
    responsibilities: list[dict],
) -> list[QualityIssue]:

    issues = []

    if not responsibilities:

        issues.append(
            QualityIssue(
                field="responsibilities",
                issue_type="empty_list",
                level="error",
                message="岗位职责列表为空",
            )
        )

        return issues

    for index, responsibility in enumerate(
        responsibilities
    ):

        # ----------------------------------------------------
        # description
        # ----------------------------------------------------

        issues.extend(
            validate_text_quality(
                value=responsibility["description"],
                field_path=(
                    f"responsibilities"
                    f"[{index}].description"
                ),
                required=True,
            )
        )

        # ----------------------------------------------------
        # tasks
        # ----------------------------------------------------
        if responsibility["description"] != "完成上级交办的其他工作":
            issues.extend(
                validate_string_list_quality(
                    values=responsibility["tasks"],
                    field_path=(
                        f"responsibilities"
                        f"[{index}].tasks"
                    ),
                    empty_level="warning",
                )
            )

    return issues


# ============================================================
# qualification 质量校验
# ============================================================

def validate_qualification_quality(
    qualification: dict,
) -> list[QualityIssue]:

    issues = []

    # --------------------------------------------------------
    # minimum_education
    # --------------------------------------------------------

    issues.extend(
        validate_text_quality(
            value=qualification[
                "minimum_education"
            ],
            field_path=(
                "qualification.minimum_education"
            ),
            required=True,
        )
    )

    # --------------------------------------------------------
    # education_background
    # --------------------------------------------------------

    issues.extend(
        validate_text_quality(
            value=qualification[
                "education_background"
            ],
            field_path=(
                "qualification.education_background"
            ),
            required=True,
        )
    )

    # --------------------------------------------------------
    # work_experience_raw
    #
    # None 是合法值。
    # 只有实际存在 str 时才进行质量检查。
    # --------------------------------------------------------

    work_experience = qualification.get(
        "work_experience_raw"
    )

    if work_experience is not None:

        issues.extend(
            validate_text_quality(
                value=work_experience,
                field_path=(
                    "qualification.work_experience_raw"
                ),
                required=False,
            )
        )

    # --------------------------------------------------------
    # competencies
    # --------------------------------------------------------

    issues.extend(
        validate_string_list_quality(
            values=qualification[
                "competencies"
            ],
            field_path=(
                "qualification.competencies"
            ),
            empty_level="warning",
        )
    )

    return issues


# ============================================================
# 单条 JD 质量校验
# ============================================================

def validate_jd_quality(
    jd: dict,
) -> list[QualityIssue]:

    issues = []

    # job_title
    issues.extend(
        validate_text_quality(
            value=jd["job_title"],
            field_path="job_title",
            required=True,
        )
    )

    # department
    issues.extend(
        validate_text_quality(
            value=jd["department"],
            field_path="department",
            required=True,
        )
    )

    # responsibilities
    issues.extend(
        validate_responsibilities_quality(
            jd["responsibilities"]
        )
    )

    # qualification
    issues.extend(
        validate_qualification_quality(
            jd["qualification"]
        )
    )

    return issues


# ============================================================
# 判断是否存在 ERROR
# ============================================================

def has_quality_error(
    issues: list[QualityIssue],
) -> bool:

    return any(
        issue.level == "error"
        for issue in issues
    )


# ============================================================
# 主程序
# ============================================================

def main():

    # --------------------------------------------------------
    # 读取已经通过结构校验的 JSON
    # --------------------------------------------------------

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(
            "输入 JSON 顶层必须为 list"
        )

    report = []

    pass_count = 0
    warning_count = 0
    error_count = 0

    # --------------------------------------------------------
    # 一条一条进行质量校验
    # --------------------------------------------------------

    for index, jd in enumerate(data):

        issues = validate_jd_quality(jd)

        has_error = has_quality_error(
            issues
        )

        has_warning = any(
            issue.level == "warning"
            for issue in issues
        )

        if has_error:
            status = "error"
            error_count += 1

        elif has_warning:
            status = "warning"
            warning_count += 1

        else:
            status = "pass"
            pass_count += 1

        # 只记录校验结果，
        # 不修改、不重新保存 JD 数据
        report.append(
            {
                "index": index,
                "status": status,
                "issues": [
                    asdict(issue)
                    for issue in issues
                ],
            }
        )

    # --------------------------------------------------------
    # 保存质量报告
    # --------------------------------------------------------

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # 输出统计
    # --------------------------------------------------------

    print("=" * 50)

    print(f"JD 总数: {len(data)}")
    print(f"完全通过: {pass_count}")
    print(f"存在 Warning: {warning_count}")
    print(f"存在 Error: {error_count}")

    print("=" * 50)


if __name__ == "__main__":
    main()