from pydantic import ConfigDict, BaseModel, Field


class ResponsibilityModel(BaseModel):
    model_config = ConfigDict(strict=True,extra="ignore")

    description: str = Field(description="岗位职责描述")
    tasks: list[str] = Field(description="岗位职责任务")

class QualificationModel(BaseModel):
    model_config = ConfigDict(strict=True,extra="ignore")

    minimum_education: str = Field(description="学历要求")
    education_background: str = Field(min_length=1,description="专业背景要求")
    work_experience_raw: str|None = Field(default=None,min_length=1,description="工作经验要求")
    competencies: list[str] = Field(default_factory=list,description="技能要求")

class JDModel(BaseModel):

    model_config = ConfigDict(strict=True,extra="ignore")

    job_title: str = Field(description="岗位名称")
    department: str = Field(description="部门名称")

    responsibilities: list[ResponsibilityModel] = Field(description="岗位职责")
    qualification: QualificationModel = Field(description="任职资格要求")


