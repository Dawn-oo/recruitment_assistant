from resume_parser import MinerUParser


parser = MinerUParser(
    model_version="vlm",
)

result = parser.parse(
    file_path=r"/app/services/resume_handle/人力资源培训专员简历_张晓婷.pdf",
    output_dir="data/mineru_results",
)

print("\n====================")
print("Markdown:")
print("====================")

print(result["markdown"])