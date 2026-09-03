from enum import Enum

class ResumeQueryType(str, Enum):
    WORK_EXPERIENCE = "work_experience"
    PROJECT_EXPERIENCE = "project_experience"
    SKILLS = "skills"