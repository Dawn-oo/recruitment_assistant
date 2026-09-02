from app.jd预处理.jd校验.jd_standard_schema import JDModel
from pydantic import ValidationError
import json
from pathlib import Path


# 结构性校验
project_root = Path(__file__).resolve().parents[1]
input_path = project_root / "jobs_raw.json"

INPUT_FILE = Path(input_path)


def load_json(file_path: Path):

    try:
        with file_path.open(
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)

    except FileNotFoundError:
        raise RuntimeError(
            f"文件不存在: {file_path}"
        )

    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"JSON 格式错误: "
            f"line={e.lineno}, "
            f"column={e.colno}, "
            f"message={e.msg}"
        )


def validate_jd_file(file_path: Path):
    data = load_json(file_path)

    valid_records = []
    invalid_records = []

    for index, item in enumerate(data):

        try:
            JDModel.model_validate(item)

            valid_records.append(item)

        except ValidationError as e:

            invalid_records.append(
                {
                    "index": index,
                    "data": item,
                    "errors": e.errors()
                }
            )

    return valid_records, invalid_records


def main():
    valid_records, invalid_records = (
        validate_jd_file(INPUT_FILE)
    )

    print("=" * 50)
    print(f"JD 总数: {len(valid_records) + len(invalid_records)}")
    print(f"校验通过: {len(valid_records)}")
    print(f"校验失败: {len(invalid_records)}")
    print("=" * 50)

    if invalid_records:
        print("\n结构异常数据：")

        for record in invalid_records:
            print(
                f"\nJD index: {record['index']}"
            )

            for error in record["errors"]:
                print(
                    f"字段: {error['loc']}, "
                    f"错误类型: {error['type']}, "
                    f"信息: {error['msg']}"
                )


if __name__ == "__main__":
    main()
