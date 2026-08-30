from job_intent_norm import JobIntentNormalizer



def print_result(case_name: str, result) -> None:
    print("=" * 60)
    print(f"测试场景: {case_name}")
    print(f"原始岗位: {result.raw_target_job_title}")
    print(f"是否多岗位: {result.is_multi_intent}")

    for index, intent in enumerate(result.intents, start=1):
        print(f"\n岗位 {index}:")
        print(f"  raw_title        = {intent.raw_title}")
        print(f"  normalized_title = {intent.normalized_title}")
        print(f"  resolution_type  = {intent.resolution_type.value}")

    print()


def main() -> None:
    normalizer = JobIntentNormalizer()

    test_cases = [
        # 1. 标准岗位
        (
            "标准岗位精确匹配",
            "项目经理",
        ),

        # 2. Alias
        (
            "Alias匹配",
            "项目实施经理",
        ),

        # 3. 多岗位
        (
            "多个岗位",
            "软件开发工程师、项目实施经理",
        ),

        # 4. 使用 / 的多岗位
        (
            "斜杠分隔多个岗位",
            "实施工程师/项目经理",
        ),

        # 5. 不应该盲目拆开的岗位
        (
            "无法确认的复合岗位",
            "Java/C++开发工程师",
        ),

        # 6. 完全未知岗位
        (
            "未知岗位",
            "AI Agent开发工程师",
        ),

        # 7. 空值
        (
            "没有填写求职岗位",
            None,
        ),
    ]

    for case_name, target_job_title in test_cases:
        result = normalizer.normalize(
            target_job_title
        )

        print_result(
            case_name,
            result,
        )


if __name__ == "__main__":
    main()