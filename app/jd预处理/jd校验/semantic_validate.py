import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


# ============================================================
# 文件配置
# ============================================================

project_root = Path(__file__).resolve().parents[1]
input_path = project_root / "jobs_raw.json"

INPUT_FILE = Path(input_path)

OUTPUT_FILE = Path(
    "data/validated/semantic_report.json"
)


# ============================================================
# Semantic Issue
# ============================================================

@dataclass
class SemanticIssue:
    field: str

    issue_type: str

    level: Literal[
        "warning",
        "high_risk"
    ]

    message: str

    value: str | None = None


# ============================================================
# 关键词配置
# ============================================================

# 岗位名称常见关键词
JOB_TITLE_KEYWORDS = {
    "工程师",
    "开发",
    "经理",
    "主管",
    "专员",
    "顾问",
    "架构师",
    "分析师",
    "设计师",
    "产品",
    "运营",
    "测试",
    "算法",
    "研究员",
    "助理",
    "负责人",
    "总监",
}


DEPARTMENT_SUFFIXES = (
    "部",
    "中心",
    "事业部",
    "研究院",
    "实验室",
    "办公室",
    "科",
    "处",
    "组",
)

# 这些词出现时，才比较强地说明它像一句职责描述
STRONG_RESPONSIBILITY_KEYWORDS = {
    "负责",
    "参与",
    "承担",
    "完成",
    "推进",
    "推动",
    "维护",
    "协调",
    "支持",
    "实施",
    "跟进",
    "协助",
    "组织",
}

# 学历相关关键词
EDUCATION_KEYWORDS = {
    "不限",
    "高中",
    "中专",
    "大专",
    "专科",
    "本科",
    "学士",
    "硕士",
    "研究生",
    "博士",
    "博士后",
    "学历",
    "学位",
}


# 专业背景关键词
EDUCATION_BACKGROUND_KEYWORDS = {
    "专业",
    "相关专业",
    "计算机",
    "软件工程",
    "信息",
    "数学",
    "统计",
    "电子",
    "通信",
    "自动化",
    "人工智能",
    "金融",
    "经济",
    "管理",
    "不限",
}


# 工作经验关键词
WORK_EXPERIENCE_KEYWORDS = {
    "经验",
    "工作经历",
    "工作经验",
    "年以上",
    "年及以上",
    "应届",
    "校招",
    "不限",
    "实习",
}


# 岗位职责倾向关键词
RESPONSIBILITY_KEYWORDS = {
    "负责",
    "参与",
    "承担",
    "完成",
    "推进",
    "推动",
    "维护",
    "开发",
    "设计",
    "建设",
    "优化",
    "管理",
    "协调",
    "支持",
    "制定",
    "实施",
    "跟进",
    "协助",
    "组织",
}


# 任职资格 / 能力倾向关键词
QUALIFICATION_KEYWORDS = {
    "熟悉",
    "掌握",
    "了解",
    "具备",
    "具有",
    "能够",
    "能力",
    "优先",
    "要求",
    "经验",
    "学历",
    "本科",
    "硕士",
    "博士",
    "专业",
    "擅长",
    "精通",
    "独立完成",
}


# ============================================================
# 基础辅助函数
# ============================================================

def normalize_for_match(text: str) -> str:

    text = text.strip()

    text = re.sub(
        r"\s+",
        "",
        text,
    )

    return text


def contains_any(
    text: str,
    keywords: set[str],
) -> bool:
    if not text:
        return False
    return any(
        keyword in text
        for keyword in keywords
    )


def count_keywords(
    text: str,
    keywords: set[str],
) -> int:

    return sum(
        1
        for keyword in keywords
        if keyword in text
    )


# ============================================================
# job_title
# ============================================================

def validate_job_title(
    value: str,
) -> list[SemanticIssue]:

    issues = []

    # 明显包含学历要求
    if contains_any(
        value,
        EDUCATION_KEYWORDS,
    ):
        issues.append(
            SemanticIssue(
                field="job_title",
                issue_type="possible_field_misalignment",
                level="high_risk",
                message="岗位名称中出现明显学历语义，可能发生字段错位",
                value=value,
            )
        )

    # 明显像职责描述
    responsibility_score = count_keywords(
        value,
        RESPONSIBILITY_KEYWORDS,
    )

    if (
        responsibility_score >= 2
        and not contains_any(
            value,
            JOB_TITLE_KEYWORDS,
        )
    ):
        issues.append(
            SemanticIssue(
                field="job_title",
                issue_type="responsibility_like_value",
                level="high_risk",
                message="岗位名称更像岗位职责描述",
                value=value,
            )
        )

    # 没有任何典型岗位词，只作为 warning
    if not contains_any(
        value,
        JOB_TITLE_KEYWORDS,
    ):
        issues.append(
            SemanticIssue(
                field="job_title",
                issue_type="unusual_job_title",
                level="warning",
                message="岗位名称未出现常见岗位名称特征，请人工确认",
                value=value,
            )
        )

    return issues


# ============================================================
# department
# ============================================================
def validate_department(
    value: str,
) -> list[SemanticIssue]:

    issues = []

    # =========================================
    # 1. 首先判断它是否具有明显的部门名称形式
    # =========================================
    try:
        looks_like_department = value.endswith(
            DEPARTMENT_SUFFIXES
        )
    except AttributeError as e:
        looks_like_department = False

    # =========================================
    # 2. 学历语义出现在 department 中
    #    这个仍然比较可疑
    # =========================================

    if contains_any(
        value,
        EDUCATION_KEYWORDS,
    ):
        issues.append(
            SemanticIssue(
                field="department",
                issue_type="possible_field_misalignment",
                level="high_risk",
                message="部门字段中出现明显学历相关语义",
                value=value,
            )
        )

    # =========================================
    # 3. 只有出现“强职责词”时才怀疑
    #
    # 软件开发部
    # 项目管理部
    # 产品设计部
    #
    # 都不会因为 开发/管理/设计 误报
    # =========================================

    if contains_any(
        value,
        STRONG_RESPONSIBILITY_KEYWORDS,
    ):
        issues.append(
            SemanticIssue(
                field="department",
                issue_type="responsibility_like_value",
                level="warning",
                message="部门字段中出现明显岗位职责表达",
                value=value,
            )
        )

    # =========================================
    # 4. 如果既没有部门后缀，也没有常见部门特征，
    #    才产生普通 warning
    # =========================================

    if (
        not looks_like_department
        and not contains_any(
            value,
            DEPARTMENT_SUFFIXES,
        )
    ):
        issues.append(
            SemanticIssue(
                field="department",
                issue_type="unusual_department_name",
                level="warning",
                message="部门名称未出现常见部门命名特征，请人工确认",
                value=value,
            )
        )

    return issues
# ============================================================
# minimum_education
# ============================================================

def validate_minimum_education(
    value: str,
) -> list[SemanticIssue]:

    issues = []

    has_education = contains_any(
        value,
        EDUCATION_KEYWORDS,
    )

    responsibility_score = count_keywords(
        value,
        RESPONSIBILITY_KEYWORDS,
    )

    if not has_education:
        level = (
            "high_risk"
            if responsibility_score > 0
            else "warning"
        )

        issues.append(
            SemanticIssue(
                field="qualification.minimum_education",
                issue_type="invalid_education_semantics",
                level=level,
                message="学历字段中没有识别到学历相关语义",
                value=value,
            )
        )

    if (
        responsibility_score >= 2
        and not has_education
    ):
        issues.append(
            SemanticIssue(
                field="qualification.minimum_education",
                issue_type="possible_field_misalignment",
                level="high_risk",
                message="学历字段内容更像岗位职责，可能发生字段错位",
                value=value,
            )
        )

    return issues


# ============================================================
# education_background
# ============================================================

def validate_education_background(
    value: str,
) -> list[SemanticIssue]:

    issues = []

    background_score = count_keywords(
        value,
        EDUCATION_BACKGROUND_KEYWORDS,
    )

    responsibility_score = count_keywords(
        value,
        RESPONSIBILITY_KEYWORDS,
    )

    if background_score == 0:
        issues.append(
            SemanticIssue(
                field="qualification.education_background",
                issue_type="unusual_education_background",
                level="warning",
                message="专业背景字段未发现明显专业或学科语义",
                value=value,
            )
        )

    if (
        responsibility_score >= 2
        and background_score == 0
    ):
        issues.append(
            SemanticIssue(
                field="qualification.education_background",
                issue_type="possible_field_misalignment",
                level="high_risk",
                message="专业背景字段更像岗位职责内容",
                value=value,
            )
        )

    return issues


# ============================================================
# work_experience_raw
# ============================================================

def validate_work_experience(
    value: str | None,
) -> list[SemanticIssue]:

    issues = []

    # None 合法
    if value is None:
        return issues

    has_experience_semantics = contains_any(
        value,
        WORK_EXPERIENCE_KEYWORDS,
    )

    # 例如：
    # 3年
    # 5 年以上
    has_year_pattern = bool(
        re.search(
            r"\d+\s*年",
            value,
        )
    )

    responsibility_score = count_keywords(
        value,
        RESPONSIBILITY_KEYWORDS,
    )

    if not (
        has_experience_semantics
        or has_year_pattern
    ):
        issues.append(
            SemanticIssue(
                field="qualification.work_experience_raw",
                issue_type="invalid_experience_semantics",
                level=(
                    "high_risk"
                    if responsibility_score > 0
                    else "warning"
                ),
                message="工作经验字段没有识别到工作经验相关语义",
                value=value,
            )
        )

    if (
        responsibility_score >= 2
        and not has_experience_semantics
        and not has_year_pattern
    ):
        issues.append(
            SemanticIssue(
                field="qualification.work_experience_raw",
                issue_type="possible_field_misalignment",
                level="high_risk",
                message="工作经验字段内容更像岗位职责",
                value=value,
            )
        )

    return issues


# ============================================================
# competencies
# ============================================================

def validate_competencies(
    values: list[str],
) -> list[SemanticIssue]:

    issues = []

    responsibility_like_count = 0

    for index, value in enumerate(values):

        responsibility_score = count_keywords(
            value,
            RESPONSIBILITY_KEYWORDS,
        )

        qualification_score = count_keywords(
            value,
            QUALIFICATION_KEYWORDS,
        )

        # 一个 competencies 项明显以职责为主
        if (
            responsibility_score >= 2
            and qualification_score == 0
        ):
            responsibility_like_count += 1

            issues.append(
                SemanticIssue(
                    field=(
                        f"qualification."
                        f"competencies[{index}]"
                    ),
                    issue_type="responsibility_like_competency",
                    level="warning",
                    message="该技能要求内容更像岗位职责",
                    value=value,
                )
            )

    # 如果整个 competencies 大部分都像职责，
    # 比单条异常更值得怀疑
    if values:

        ratio = (
            responsibility_like_count
            / len(values)
        )

        if (
            len(values) >= 2
            and ratio >= 0.6
        ):
            issues.append(
                SemanticIssue(
                    field="qualification.competencies",
                    issue_type="possible_field_misalignment",
                    level="high_risk",
                    message=(
                        "competencies 中超过 60% 的内容"
                        "更像岗位职责，疑似字段整体错位"
                    ),
                )
            )

    return issues


# ============================================================
# responsibilities
# ============================================================

def validate_responsibilities(
    responsibilities: list[dict],
) -> list[SemanticIssue]:

    issues = []

    texts = []

    for index, responsibility in enumerate(
        responsibilities
    ):

        description = responsibility[
            "description"
        ]

        texts.append(
            (
                f"responsibilities[{index}].description",
                description,
            )
        )

        for task_index, task in enumerate(
            responsibility["tasks"]
        ):
            texts.append(
                (
                    (
                        f"responsibilities[{index}]"
                        f".tasks[{task_index}]"
                    ),
                    task,
                )
            )

    qualification_like_count = 0

    for field_path, value in texts:

        responsibility_score = count_keywords(
            value,
            RESPONSIBILITY_KEYWORDS,
        )

        qualification_score = count_keywords(
            value,
            QUALIFICATION_KEYWORDS,
        )

        # 明显学历语义出现在职责里
        has_education = contains_any(
            value,
            EDUCATION_KEYWORDS,
        )

        has_experience = (
            contains_any(
                value,
                WORK_EXPERIENCE_KEYWORDS,
            )
            or bool(
                re.search(
                    r"\d+\s*年",
                    value,
                )
            )
        )

        if (
            has_education
            or has_experience
        ):
            qualification_like_count += 1

            issues.append(
                SemanticIssue(
                    field=field_path,
                    issue_type="qualification_like_responsibility",
                    level="warning",
                    message="岗位职责中出现明显学历或工作经验要求",
                    value=value,
                )
            )

        elif (
            qualification_score >= 2
            and responsibility_score == 0
        ):
            qualification_like_count += 1

            issues.append(
                SemanticIssue(
                    field=field_path,
                    issue_type="qualification_like_responsibility",
                    level="warning",
                    message="该岗位职责更像任职资格要求",
                    value=value,
                )
            )

    # responsibilities 大面积错位
    if texts:

        ratio = (
            qualification_like_count
            / len(texts)
        )

        if (
            len(texts) >= 3
            and ratio >= 0.6
        ):
            issues.append(
                SemanticIssue(
                    field="responsibilities",
                    issue_type="possible_field_misalignment",
                    level="high_risk",
                    message=(
                        "responsibilities 中超过 60% "
                        "的内容更像任职资格，疑似字段整体错位"
                    ),
                )
            )

    return issues


# ============================================================
# 跨字段重复检测
# ============================================================

def validate_cross_field_overlap(
    jd: dict,
) -> list[SemanticIssue]:

    """
    检查同一段文本是否同时出现在不同语义字段中。

    这通常不是绝对错误，
    但可能说明 Parser / Cleaner 发生了字段复制或错位。
    """

    issues = []

    responsibility_values = []

    for index, responsibility in enumerate(
        jd["responsibilities"]
    ):

        responsibility_values.append(
            (
                (
                    f"responsibilities"
                    f"[{index}].description"
                ),
                responsibility["description"],
            )
        )

        for task_index, task in enumerate(
            responsibility["tasks"]
        ):

            responsibility_values.append(
                (
                    (
                        f"responsibilities"
                        f"[{index}]"
                        f".tasks[{task_index}]"
                    ),
                    task,
                )
            )

    qualification_values = [
        (
            "qualification.minimum_education",
            jd["qualification"][
                "minimum_education"
            ],
        ),
        (
            "qualification.education_background",
            jd["qualification"][
                "education_background"
            ],
        ),
    ]

    work_experience = jd[
        "qualification"
    ].get(
        "work_experience_raw"
    )

    if work_experience is not None:

        qualification_values.append(
            (
                "qualification.work_experience_raw",
                work_experience,
            )
        )

    for index, competency in enumerate(
        jd["qualification"]["competencies"]
    ):

        qualification_values.append(
            (
                (
                    f"qualification."
                    f"competencies[{index}]"
                ),
                competency,
            )
        )

    # 职责 vs 任职要求完全重复
    for responsibility_path, responsibility_text in (
        responsibility_values
    ):

        responsibility_normalized = (
            normalize_for_match(
                responsibility_text
            )
        )

        if not responsibility_normalized:
            continue

        for qualification_path, qualification_text in (
            qualification_values
        ):

            qualification_normalized = (
                normalize_for_match(
                    qualification_text
                )
            )

            if (
                responsibility_normalized
                == qualification_normalized
            ):

                issues.append(
                    SemanticIssue(
                        field=responsibility_path,
                        issue_type="cross_field_duplicate",
                        level="warning",
                        message=(
                            f"与 {qualification_path} "
                            f"内容完全相同，可能存在字段复制或错位"
                        ),
                        value=responsibility_text,
                    )
                )

    return issues


# ============================================================
# 单条 JD 总语义校验
# ============================================================

def validate_jd_semantics(
    jd: dict,
) -> list[SemanticIssue]:

    issues = []

    issues.extend(
        validate_job_title(
            jd["job_title"]
        )
    )

    issues.extend(
        validate_department(
            jd["department"]
        )
    )

    qualification = jd[
        "qualification"
    ]

    issues.extend(
        validate_minimum_education(
            qualification[
                "minimum_education"
            ]
        )
    )

    issues.extend(
        validate_education_background(
            qualification[
                "education_background"
            ]
        )
    )

    issues.extend(
        validate_work_experience(
            qualification.get(
                "work_experience_raw"
            )
        )
    )

    issues.extend(
        validate_competencies(
            qualification[
                "competencies"
            ]
        )
    )

    issues.extend(
        validate_responsibilities(
            jd[
                "responsibilities"
            ]
        )
    )

    issues.extend(
        validate_cross_field_overlap(
            jd
        )
    )

    return issues


# ============================================================
# 判断状态
# ============================================================

def get_semantic_status(
    issues: list[SemanticIssue],
) -> str:

    if any(
        issue.level == "high_risk"
        for issue in issues
    ):
        return "high_risk"

    if issues:
        return "warning"

    return "pass"


# ============================================================
# 主程序
# ============================================================

def main():

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(
            "输入 JSON 顶层必须是 list"
        )

    report = []

    pass_count = 0
    warning_count = 0
    high_risk_count = 0

    for index, jd in enumerate(data):

        issues = validate_jd_semantics(
            jd
        )

        status = get_semantic_status(
            issues
        )

        if status == "pass":
            pass_count += 1

        elif status == "warning":
            warning_count += 1

        else:
            high_risk_count += 1

        report.append(
            {
                "index": index,

                "job_title": jd.get(
                    "job_title"
                ),

                "department": jd.get(
                    "department"
                ),

                "status": status,

                "issues": [
                    asdict(issue)
                    for issue in issues
                ],
            }
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 60)

    print(
        f"JD 总数: {len(data)}"
    )

    print(
        f"语义校验完全通过: {pass_count}"
    )

    print(
        f"存在 Warning: {warning_count}"
    )

    print(
        f"存在 High Risk: {high_risk_count}"
    )

    print(
        f"报告路径: {OUTPUT_FILE}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()