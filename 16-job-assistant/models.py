"""
AI 简历生成器 - 数据模型
========================
定义所有 Pydantic 结构化数据模型，用于：
- 约束 LLM 输出（通过 PydanticOutputParser）
- 在函数间传递类型安全的数据
"""

from pydantic import BaseModel, Field
from typing import List


# ============================================================
# 用户画像（内容底线——不能造假）
# ============================================================

class WorkExperience(BaseModel):
    """单段工作/项目经历"""
    company: str = Field(description="公司或项目名称")
    title: str = Field(description="职位名称")
    duration: str = Field(description="在职时间段，如 '2022.06-2025.03'")
    achievements: List[str] = Field(description="主要成就或职责，每条一项")


class UserProfile(BaseModel):
    """用户画像——简历的「内容底线」，所有生成不能超出这个范围"""
    name: str = Field(description="候选人姓名")
    contact: str = Field(description="联系方式，包括邮箱和电话")
    skills: List[str] = Field(description="技能列表，如 ['Python', 'FastAPI', 'Docker']")
    experience: List[WorkExperience] = Field(description="工作/项目经历列表")
    education: str = Field(description="教育背景，如 '南京大学 计算机科学 本科 2016-2020'")


# ============================================================
# 风格画像（从样本简历提取——怎么写得好看）
# ============================================================

class StyleProfile(BaseModel):
    """风格画像——样本简历的写法特征"""
    structure: str = Field(
        description="简历的章节顺序和标题，如 '个人信息 → 技能 → 工作经历 → 教育背景'"
    )
    tone: str = Field(
        description="措辞风格，如 '简洁专业，用数据说话' 或 '偏技术术语，动词开头'"
    )
    format_patterns: str = Field(
        description="可复用的句式模板，如 '用 STAR 法则描述经历' 或 '每段经历 3-4 条成就'"
    )


# ============================================================
# JD 要求（优化方向——往哪使劲）
# ============================================================

class JDRequirements(BaseModel):
    """JD 结构化要求"""
    title: str = Field(description="岗位名称")
    must_have: List[str] = Field(description="必备要求，如 ['5年Python经验', '本科以上']")
    nice_to_have: List[str] = Field(description="加分项，如 ['有大厂经验优先', '开源贡献']")
    keywords: List[str] = Field(description="JD 中反复出现的关键词，如 ['高并发', '微服务', 'K8s']")
    hidden_preferences: str = Field(description="从措辞推断的隐性偏好，如 '偏好有大厂背景的候选人'")
