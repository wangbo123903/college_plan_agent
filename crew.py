import os
import sys
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv

# Windows utf8编码
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise Exception("请在.env文件配置 DASHSCOPE_API_KEY")

# ===== CrewAI原生LLM，对接通义千问OpenAI兼容接口 =====
llm = LLM(
    model="openai/qwen-turbo",
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.7
)

# ========== Agent定义 ==========
profile_agent = Agent(
    role="学生画像分析师",
    goal="根据用户填写的专业、技能、目标，生成清晰的学生画像",
    backstory="你是一位资深教育规划师，擅长快速分析大学生的背景和目标。",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

path_agent = Agent(
    role="成长轨迹匹配专家",
    goal="根据学生画像，匹配最合适的学长学姐成长路径，并给出理由",
    backstory="你拥有大量真实学长案例库经验，擅长找到相似度最高的成功路径。",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

goal_agent = Agent(
    role="目标拆解规划师",
    goal="把用户的大目标拆解成可执行的阶段性目标和每日待办",
    backstory="你是目标管理专家，擅长把宏大目标变成每天能完成的小任务。",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# 2. 定义任务
def create_tasks(user_input: dict):
    profile_task = Task(
        description=f"""
        分析以下用户信息，生成结构化学生画像：
        专业：{user_input.get('major')}
        年级：{user_input.get('grade')}
        技能：{user_input.get('skills')}
        目标：{user_input.get('goal')}
        城市：{user_input.get('city', '未知')}
        输出格式：清晰的画像描述 + 关键标签。
        """,
        expected_output="一份简洁的学生画像报告",
        agent=profile_agent
    )

    path_task = Task(
        description=f"""
        根据上一步的画像，匹配最合适的成长轨迹。
        用户目标是：{user_input.get('goal')}
        请给出：
        1. 最匹配的学长类型
        2. 匹配相似度（百分比）
        3. 为什么匹配的理由（3点）
        4. 推荐的核心行动路径
        """,
        expected_output="匹配结果 + 理由 + 行动路径",
        agent=path_agent,
        context=[profile_task]
    )

    goal_task = Task(
        description=f"""
        根据匹配结果，把用户目标拆解成：
        1. 3个阶段性目标（带截止日期建议）
        2. 本周可执行的5个待办事项
        3. 今日重点任务1个
        用户目标：{user_input.get('goal')}
        """,
        expected_output="结构化的目标拆解方案",
        agent=goal_agent,
        context=[path_task]
    )
    return [profile_task, path_task, goal_task]

# 3. 执行Crew
def run_crew(user_input: dict):
    print("开始执行CrewAI多Agent任务，输入：", user_input)
    tasks = create_tasks(user_input)
    crew = Crew(
        agents=[profile_agent, path_agent, goal_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        share_crew = False
    )
    try:
        result = crew.kickoff(inputs=user_input)
        return str(result.raw)
    except Exception as e:
        print(f"Crew执行异常：{e}")
        return f"任务异常：{str(e)}"
