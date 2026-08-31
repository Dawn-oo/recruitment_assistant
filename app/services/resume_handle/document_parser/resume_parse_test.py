from resume_parser import MinerUParser


parser = MinerUParser(
    model_version="vlm",
)

result = parser.parse(
    file_path=r"E:\Project\assistant_for_recruitment\app\services\resume_handle\人力资源培训专员简历_张晓婷.pdf"
)


print("\n====================")
print("Markdown:")
print("====================")

print(result.markdown)
