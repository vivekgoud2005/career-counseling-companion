"""
Career Counseling Companion
Multi-agent AI application powered by IBM Granite on watsonx.ai
"""

import os
import json
import logging
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IBM watsonx.ai initialisation
# ---------------------------------------------------------------------------
try:
    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

    _WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", "")
    _WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    _PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
    _MODEL_ID = os.getenv("GRANITE_MODEL_ID", "ibm/granite-4-h-small")

    # Treat unedited placeholder values as "not set"
    _PLACEHOLDER_VALUES = {
        "your_ibm_cloud_api_key_here",
        "your_watsonx_project_id_here",
        "",
    }
    _credentials_valid = (
        _WATSONX_API_KEY not in _PLACEHOLDER_VALUES
        and _PROJECT_ID not in _PLACEHOLDER_VALUES
    )

    if _credentials_valid:
        _credentials = Credentials(url=_WATSONX_URL, api_key=_WATSONX_API_KEY)
        _client = APIClient(_credentials)
        _model = ModelInference(
            model_id=_MODEL_ID,
            api_client=_client,
            project_id=_PROJECT_ID,
            params={
                GenParams.MAX_NEW_TOKENS: 1024,
                GenParams.TEMPERATURE: 0.3,
                GenParams.TOP_P: 0.9,
                GenParams.REPETITION_PENALTY: 1.1,
            },
        )
        WATSONX_AVAILABLE = True
        logger.info("watsonx.ai client initialised with model %s", _MODEL_ID)
    else:
        WATSONX_AVAILABLE = False
        _MODEL_ID = os.getenv("GRANITE_MODEL_ID", "ibm/granite-3-2-8b-instruct")
        logger.warning(
            "watsonx.ai credentials not set or contain placeholder values – "
            "running in demo/mock mode. Set WATSONX_API_KEY and "
            "WATSONX_PROJECT_ID in .env to enable live inference."
        )
except Exception as exc:  # pragma: no cover
    WATSONX_AVAILABLE = False
    _MODEL_ID = os.getenv("GRANITE_MODEL_ID", "ibm/granite-3-2-8b-instruct")
    logger.warning("watsonx.ai init failed: %s – running in demo/mock mode", exc)


def _generate(prompt: str, max_tokens: int = 1024, student_name: str = "Student") -> str:
    """Call IBM Granite via watsonx.ai, with a mock fallback for local dev."""
    if WATSONX_AVAILABLE:
        try:
            result = _model.generate_text(prompt=prompt)
            return result.strip()
        except Exception as exc:
            logger.error("Generation error: %s", exc)
            return _mock_generate(prompt, student_name)
    return _mock_generate(prompt, student_name)


# ---------------------------------------------------------------------------
# Per-agent mock functions  (used when watsonx credentials are absent)
# Each function receives the already-assembled prompt and extracts key details
# from it so the mock output is personalised to the actual student input.
# ---------------------------------------------------------------------------

def _mock_academic(prompt: str) -> str:
    """Return correctly-formatted mock output for the Academic Analysis agent."""
    # Extract the highest and lowest subjects from the prompt text
    import re
    subject_pcts = re.findall(r"- (.+?): \d+/\d+ \((\d+)%\)", prompt)
    if subject_pcts:
        sorted_subj = sorted(subject_pcts, key=lambda x: int(x[1]), reverse=True)
        top = [s[0] for s in sorted_subj[:2]]
        bottom = [s[0] for s in sorted_subj[-2:]]
    else:
        top = ["Mathematics", "Science"]
        bottom = ["English", "Social Studies"]

    strengths = ", ".join(top + ["Analytical reasoning"])
    weaknesses = ", ".join(bottom + ["Written communication"])
    return (
        f"STRENGTHS: {strengths}\n"
        f"WEAKNESSES: {weaknesses}\n"
        f"OVERALL_PROFILE: Strong STEM-oriented learner with quantitative aptitude and analytical mindset."
    )


def _mock_aptitude(prompt: str) -> str:
    """Return correctly-formatted mock output for the Interest & Aptitude agent."""
    import re
    interests_m = re.search(r"Interests: (.+)", prompt)
    skills_m    = re.search(r"Current Skills: (.+)", prompt)
    raw_interests = interests_m.group(1).strip() if interests_m else "Technology"
    raw_skills    = skills_m.group(1).strip()    if skills_m    else "Problem-solving"
    # Derive RIASEC themes from listed interests
    themes = "Investigative, Realistic"
    if any(w in raw_interests.lower() for w in ["art", "design", "music", "creative"]):
        themes = "Artistic, Investigative"
    elif any(w in raw_interests.lower() for w in ["people", "teach", "social", "help"]):
        themes = "Social, Enterprising"
    elif any(w in raw_interests.lower() for w in ["business", "manage", "lead", "market"]):
        themes = "Enterprising, Conventional"
    return (
        f"INTERESTS: {themes}, Conventional\n"
        f"APTITUDE_AREAS: Software Engineering, Data Science, Research & Development, "
        f"Systems Analysis\n"
        f"PERSONALITY_SUMMARY: You are a curious, detail-oriented thinker with a strong "
        f"affinity for {raw_interests}. Your skills in {raw_skills} suggest a natural fit "
        f"for roles that combine technical precision with creative problem-solving. You "
        f"thrive in structured environments that reward depth of expertise."
    )


def _mock_market(prompt: str) -> str:
    """Return correctly-formatted mock output for the Labor Market agent."""
    return (
        "TOP_TRENDS: AI/ML engineering demand up 35% year-over-year, "
        "Cloud computing roles growing at 28% annually, "
        "Cybersecurity workforce gap exceeds 3.5 million globally, "
        "Data science remains a top-5 most in-demand occupation\n"
        "IN_DEMAND_SKILLS: Python, Cloud Platforms (AWS/Azure/GCP), "
        "Machine Learning, SQL & Data Engineering, DevOps & CI/CD, "
        "Prompt Engineering\n"
        "EMERGING_ROLES: AI/ML Engineer, Prompt Engineer, Cloud Solutions Architect\n"
        "MARKET_INSIGHTS: The technology sector continues to outpace all other industries "
        "in job creation. Professionals with AI literacy and cloud skills command a 20–40% "
        "salary premium over peers. Building a portfolio of practical projects and "
        "earning 1–2 recognised certifications will significantly accelerate your entry "
        "into the job market. Remote-first hiring has expanded the talent pool globally, "
        "making skills and portfolio quality the primary differentiators."
    )


def _mock_recommendation(prompt: str, student_name: str = "Student") -> str:
    """Return correctly-formatted mock output for the Career Recommendation agent."""
    return (
        f"CAREER_1: Data Scientist | SCORE: 93 | "
        f"FIT: Your STEM academic strengths and analytical personality align perfectly with data science. "
        f"The field is the fastest-growing STEM role with an average salary of $125,000. | "
        f"SKILLS_NEEDED: Python & Pandas, Statistical Modeling, Machine Learning (scikit-learn) | "
        f"NEXT_STEP: Complete the 'Machine Learning Specialization' by Andrew Ng on Coursera\n"

        f"CAREER_2: AI / ML Engineer | SCORE: 88 | "
        f"FIT: Strong mathematics background and tech interests make this a natural fit. "
        f"AI engineering offers premium compensation ($140k–$200k) and is at the forefront of innovation. | "
        f"SKILLS_NEEDED: Deep Learning (PyTorch/TensorFlow), MLOps & Model Deployment, Cloud Platforms | "
        f"NEXT_STEP: Build and deploy an end-to-end ML project on GitHub using a public dataset\n"

        f"CAREER_3: Cybersecurity Analyst | SCORE: 79 | "
        f"FIT: Your problem-solving aptitude and detail-oriented nature suit threat analysis well. "
        f"With 3.5 million unfilled positions globally, job security is exceptionally strong. | "
        f"SKILLS_NEEDED: Network Security Fundamentals, Python Scripting for Security, CompTIA Security+ | "
        f"NEXT_STEP: Earn the CompTIA Security+ certification (beginner-friendly, widely recognised)\n"

        f"CAREER_4: Cloud Solutions Architect | SCORE: 74 | "
        f"FIT: Analytical thinking translates well to designing resilient cloud infrastructure. "
        f"Average salary exceeds $155,000 and demand is accelerating across every industry. | "
        f"SKILLS_NEEDED: AWS or Azure Fundamentals, System Design & Architecture, Terraform/IaC | "
        f"NEXT_STEP: Earn the AWS Cloud Practitioner certification as your first cloud credential\n"

        f"SKILL_GAPS: Cloud Infrastructure & DevOps, Advanced Statistical Methods, "
        f"Version Control (Git/GitHub), Professional Communication, "
        f"Database Engineering (SQL & NoSQL)\n"

        f"NEXT_STEPS: "
        f"Enroll in 'Machine Learning Specialization' on Coursera (Andrew Ng) – free to audit;"
        f"Obtain the AWS Cloud Practitioner or Google Associate Cloud Engineer certification;"
        f"Build a capstone portfolio project on GitHub and share it on LinkedIn\n"

        f"SUMMARY: {student_name}, your profile reflects a strong STEM foundation paired with "
        f"genuine curiosity and analytical drive — a combination that is highly sought after in "
        f"today's technology-led job market. Data Science and AI Engineering stand out as your "
        f"best-matched pathways, offering both intellectual fulfilment and exceptional long-term "
        f"earning potential. By closing the identified skill gaps through focused online learning "
        f"and hands-on projects over the next 12–18 months, you will be well-positioned to land "
        f"competitive roles at leading technology companies."
    )


def _mock_generate(prompt: str, student_name: str = "Student") -> str:
    """
    Route mock generation to the correct per-agent function based on prompt content.
    Each function returns a properly-structured, multi-line response that the calling
    agent's parser can successfully extract tokens from.
    """
    p = prompt.lower()
    if "academic performance analyst" in p or "subject performance" in p:
        return _mock_academic(prompt)
    if "career psychologist" in p or "riasec" in p:
        return _mock_aptitude(prompt)
    if "labor market intelligence" in p or "retrieved labor market" in p:
        return _mock_market(prompt)
    # Default: recommendation agent
    return _mock_recommendation(prompt, student_name)


# ---------------------------------------------------------------------------
# RAG – Career & Labour Market Knowledge Base
# ---------------------------------------------------------------------------
CAREER_CORPUS = [
    # Technology
    "Software engineers design, develop, and maintain software systems. High demand for Python, JavaScript, and cloud skills. Average US salary $130,000. 25% growth projected by 2032.",
    "Data scientists analyze large datasets using ML and statistics to derive business insights. Key skills: Python, R, SQL, TensorFlow, PyTorch. Average salary $125,000. Fastest-growing STEM role.",
    "AI/ML engineers build and deploy machine learning models at scale. Skills needed: deep learning, MLOps, cloud platforms. Salary range $140,000–$200,000.",
    "Cybersecurity analysts protect organizational assets from digital threats. Certifications: CISSP, CEH, CompTIA Security+. 3.5 million unfilled positions globally. Salary $110,000.",
    "Cloud architects design and oversee cloud computing strategies. AWS, Azure, GCP expertise essential. Average salary $155,000.",
    "DevOps engineers bridge development and operations using CI/CD, Docker, Kubernetes. High demand in startups and enterprises alike.",
    "Full-stack developers build both front-end and back-end web applications. React, Node.js, databases. Entry-level salary $85,000.",
    # Healthcare
    "Biomedical engineers develop medical devices and equipment. Strong math and biology background required. Average salary $97,000. Growing with aging population.",
    "Healthcare data analysts apply analytics to patient data to improve outcomes. Skills: SQL, Python, HL7/FHIR standards. Bridge between IT and medicine.",
    "Genetic counselors guide patients on inherited conditions using genomics. Requires biology, psychology, and communication skills. Master's degree typically required.",
    # Finance
    "Financial analysts evaluate investments and economic trends. CFA certification valued. Excel, Python, financial modeling skills essential. Average salary $96,000.",
    "FinTech engineers build payment systems, blockchain apps, and robo-advisors. Python, Solidity, cloud platforms in demand.",
    "Quantitative analysts (quants) use mathematical models for trading strategies. Requires advanced math, statistics, and C++/Python. High compensation in hedge funds.",
    # Creative & Design
    "UX/UI designers create intuitive digital experiences. Figma, Adobe XD, user research skills. Growing demand in tech and e-commerce. Average salary $95,000.",
    "Digital marketing specialists manage SEO, SEM, social media, and analytics. Google Ads, Meta Ads, HubSpot proficiency required.",
    # Science & Research
    "Environmental scientists study environmental problems and develop solutions. Growing due to climate change initiatives. Data analysis and field research skills needed.",
    "Research scientists conduct experiments and publish findings. PhD typically required. Pharmaceutical and biotech sectors highly competitive.",
    # Business
    "Product managers define product vision and roadmap, collaborating with engineering, design, and marketing. MBA or STEM background valued.",
    "Management consultants advise organizations on strategy and operations. Top firms: McKinsey, BCG, Bain. Strong analytical and communication skills required.",
    "Supply chain managers optimize logistics networks. ERP systems (SAP, Oracle), data analytics, operations research skills needed.",
    # Labor market trends
    "2024 labor market: AI skills command 20–40% salary premium. Remote work normalized in tech. Green energy jobs growing 11% annually.",
    "In-demand certifications 2024: AWS Solutions Architect, Google Professional Data Engineer, Certified Kubernetes Administrator, PMP, CFA.",
    "Skill gaps identified globally: AI literacy, cybersecurity, data analysis, critical thinking, cross-cultural communication.",
    "STEM graduates earn 26% more than non-STEM graduates on average over a career lifetime.",
    "Top industries hiring 2024–2026: Healthcare tech, renewable energy, AI/automation, cybersecurity, cloud computing.",
    "Automation risk: routine manual and clerical jobs declining; creative, interpersonal, and technical roles expanding.",
    "Emerging roles: Prompt engineers, AI ethicists, quantum computing researchers, climate data scientists, longevity researchers.",
]

# Build FAISS index at startup
try:
    import faiss
    from sentence_transformers import SentenceTransformer

    _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    _corpus_embeddings = _embedder.encode(CAREER_CORPUS, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(_corpus_embeddings)
    _faiss_dim = _corpus_embeddings.shape[1]
    _faiss_index = faiss.IndexFlatIP(_faiss_dim)  # Inner-product = cosine after L2-norm
    _faiss_index.add(_corpus_embeddings)
    RAG_AVAILABLE = True
    logger.info("FAISS index built: %d documents, dim=%d", len(CAREER_CORPUS), _faiss_dim)
except Exception as exc:
    RAG_AVAILABLE = False
    logger.warning("FAISS/sentence-transformers unavailable: %s – RAG disabled", exc)


def rag_retrieve(query: str, top_k: int = 5) -> str:
    """Retrieve top-k relevant career corpus passages for a query."""
    if not RAG_AVAILABLE:
        return "\n".join(CAREER_CORPUS[:top_k])
    query_vec = _embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)
    distances, indices = _faiss_index.search(query_vec, top_k)
    results = [CAREER_CORPUS[i] for i in indices[0] if i < len(CAREER_CORPUS)]
    return "\n".join(results)


# ---------------------------------------------------------------------------
# Agent 1 – Academic Analysis Agent
# ---------------------------------------------------------------------------
def academic_analysis_agent(academic_data: dict) -> dict:
    """
    Analyses student academic performance and returns structured strengths/weaknesses.
    Input:  { subjects: [{name, marks, max_marks}], grade_level, gpa }
    Output: { strengths, weaknesses, overall_profile, subject_scores }
    """
    subjects = academic_data.get("subjects", [])
    grade_level = academic_data.get("grade_level", "12th Grade")
    gpa = academic_data.get("gpa", "")

    subject_lines = "\n".join(
        f"- {s['name']}: {s['marks']}/{s.get('max_marks', 100)} "
        f"({round(s['marks'] / s.get('max_marks', 100) * 100)}%)"
        for s in subjects
    )

    prompt = f"""You are an academic performance analyst. Analyse the following student academic data and provide:
1. Top 3 STRENGTHS (academic subjects and skills they excel at)
2. Top 3 WEAKNESSES (areas needing improvement)
3. OVERALL_PROFILE in one sentence

Student Grade Level: {grade_level}
GPA/CGPA: {gpa}
Subject Performance:
{subject_lines}

Respond in this exact format:
STRENGTHS: <comma-separated list>
WEAKNESSES: <comma-separated list>
OVERALL_PROFILE: <one sentence>"""

    raw = _generate(prompt)

    # Parse structured output
    strengths, weaknesses, profile = [], [], ""
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("STRENGTHS:"):
            strengths = [s.strip() for s in line.replace("STRENGTHS:", "").split(",")]
        elif line.startswith("WEAKNESSES:"):
            weaknesses = [w.strip() for w in line.replace("WEAKNESSES:", "").split(",")]
        elif line.startswith("OVERALL_PROFILE:"):
            profile = line.replace("OVERALL_PROFILE:", "").strip()

    subject_scores = {
        s["name"]: round(s["marks"] / s.get("max_marks", 100) * 100)
        for s in subjects
    }

    return {
        "strengths": strengths or ["Strong academic background"],
        "weaknesses": weaknesses or ["Areas identified for improvement"],
        "overall_profile": profile or "Well-rounded student with balanced performance.",
        "subject_scores": subject_scores,
    }


# ---------------------------------------------------------------------------
# Agent 2 – Interest & Aptitude Agent
# ---------------------------------------------------------------------------
def interest_aptitude_agent(interest_data: dict) -> dict:
    """
    Interprets student interests, skills, and personality from questionnaire.
    Input:  { interests, skills, personality_traits, hobbies, work_style }
    Output: { interpreted_interests, aptitude_areas, personality_summary }
    """
    interests = ", ".join(interest_data.get("interests", []))
    skills = ", ".join(interest_data.get("skills", []))
    traits = ", ".join(interest_data.get("personality_traits", []))
    hobbies = interest_data.get("hobbies", "")
    work_style = interest_data.get("work_style", "")

    prompt = f"""You are a career psychologist specialising in interest and aptitude assessment. Analyse the following student profile:

Interests: {interests}
Current Skills: {skills}
Personality Traits: {traits}
Hobbies: {hobbies}
Preferred Work Style: {work_style}

Provide:
1. INTERESTS: Interpreted core interest themes (e.g., Investigative, Artistic, Social – RIASEC)
2. APTITUDE_AREAS: 3-5 specific career aptitude areas this student is suited for
3. PERSONALITY_SUMMARY: One paragraph summarising their personality and its career implications

Respond in this format:
INTERESTS: <comma-separated themes>
APTITUDE_AREAS: <comma-separated areas>
PERSONALITY_SUMMARY: <one paragraph>"""

    raw = _generate(prompt)

    interests_out, aptitude_areas, personality_summary = [], [], ""
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("INTERESTS:"):
            interests_out = [i.strip() for i in line.replace("INTERESTS:", "").split(",")]
        elif line.startswith("APTITUDE_AREAS:"):
            aptitude_areas = [a.strip() for a in line.replace("APTITUDE_AREAS:", "").split(",")]
        elif line.startswith("PERSONALITY_SUMMARY:"):
            personality_summary = line.replace("PERSONALITY_SUMMARY:", "").strip()

    return {
        "interpreted_interests": interests_out or [interests],
        "aptitude_areas": aptitude_areas or ["Technology", "Research", "Analysis"],
        "personality_summary": personality_summary or "Curious and driven individual suited for analytical work.",
    }


# ---------------------------------------------------------------------------
# Agent 3 – Labor Market Trends Agent  (RAG-powered)
# ---------------------------------------------------------------------------
def labor_market_agent(career_context: str) -> dict:
    """
    Retrieves real-time labor market trends relevant to the student's profile using RAG.
    Input:  career_context (free-form description of academic + interest profile)
    Output: { top_trends, in_demand_skills, emerging_roles, market_insights }
    """
    rag_context = rag_retrieve(career_context, top_k=6)

    prompt = f"""You are a labor market intelligence analyst. Using the retrieved market data below, provide current trends relevant to a student with the described profile.

Student Profile Context: {career_context}

Retrieved Labor Market Data:
{rag_context}

Based on this data, provide:
TOP_TRENDS: 3-4 most relevant labor market trends for this student (comma-separated)
IN_DEMAND_SKILLS: 5-6 skills currently most in demand (comma-separated)
EMERGING_ROLES: 2-3 emerging job roles to watch (comma-separated)
MARKET_INSIGHTS: One paragraph of actionable market intelligence

Respond in this format:
TOP_TRENDS: <comma-separated>
IN_DEMAND_SKILLS: <comma-separated>
EMERGING_ROLES: <comma-separated>
MARKET_INSIGHTS: <paragraph>"""

    raw = _generate(prompt)

    trends, skills, roles, insights = [], [], [], ""
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("TOP_TRENDS:"):
            trends = [t.strip() for t in line.replace("TOP_TRENDS:", "").split(",")]
        elif line.startswith("IN_DEMAND_SKILLS:"):
            skills = [s.strip() for s in line.replace("IN_DEMAND_SKILLS:", "").split(",")]
        elif line.startswith("EMERGING_ROLES:"):
            roles = [r.strip() for r in line.replace("EMERGING_ROLES:", "").split(",")]
        elif line.startswith("MARKET_INSIGHTS:"):
            insights = line.replace("MARKET_INSIGHTS:", "").strip()

    return {
        "top_trends": trends or ["AI adoption accelerating", "Remote work normalised"],
        "in_demand_skills": skills or ["Python", "Cloud", "Machine Learning", "SQL", "Communication"],
        "emerging_roles": roles or ["AI Engineer", "Data Scientist", "Cloud Architect"],
        "market_insights": insights or "Technology roles offer strong growth and compensation prospects.",
        "rag_sources_used": len(CAREER_CORPUS),
    }


# ---------------------------------------------------------------------------
# Agent 4 – Career Pathway Recommendation Agent
# ---------------------------------------------------------------------------
def career_recommendation_agent(
    academic: dict,
    aptitude: dict,
    market: dict,
    student_name: str = "Student",
) -> dict:
    """
    Synthesises outputs from all three agents into personalised career recommendations.
    Returns ranked career paths, skill-gap analysis, and next steps.
    """
    rag_context = rag_retrieve(
        " ".join(aptitude.get("aptitude_areas", [])) + " " + " ".join(market.get("in_demand_skills", [])),
        top_k=5,
    )

    prompt = f"""You are a senior career counselor. Using the complete student analysis below, generate personalised, ranked career pathway recommendations.

=== ACADEMIC PROFILE ===
Strengths: {", ".join(academic.get("strengths", []))}
Weaknesses: {", ".join(academic.get("weaknesses", []))}
Overall: {academic.get("overall_profile", "")}

=== INTEREST & APTITUDE PROFILE ===
Interest Themes: {", ".join(aptitude.get("interpreted_interests", []))}
Aptitude Areas: {", ".join(aptitude.get("aptitude_areas", []))}
Personality: {aptitude.get("personality_summary", "")}

=== LABOR MARKET CONTEXT ===
Top Trends: {", ".join(market.get("top_trends", []))}
In-Demand Skills: {", ".join(market.get("in_demand_skills", []))}
Emerging Roles: {", ".join(market.get("emerging_roles", []))}
Market Insights: {market.get("market_insights", "")}

=== RELEVANT CAREER DATA (RAG Retrieved) ===
{rag_context}

Generate exactly 4 ranked career pathways. For each provide:
- Career title
- Match score (0-100)
- Why it fits (2 sentences)
- Required skills to develop (3 items)
- Recommended next step

Also provide:
SKILL_GAPS: List 5 specific skills the student needs to develop (comma-separated)
NEXT_STEPS: List 3 concrete action items (courses, certifications, projects) (semicolon-separated)
SUMMARY: One encouraging paragraph summarising their career outlook

Format each pathway as:
CAREER_1: <title> | SCORE: <n> | FIT: <reason> | SKILLS_NEEDED: <s1,s2,s3> | NEXT_STEP: <step>
CAREER_2: ...
CAREER_3: ...
CAREER_4: ...
SKILL_GAPS: <comma-separated>
NEXT_STEPS: <semicolon-separated>
SUMMARY: <paragraph>"""

    raw = _generate(prompt, student_name=student_name)

    careers = []
    skill_gaps = []
    next_steps = []
    summary = ""

    for line in raw.splitlines():
        line = line.strip()
        for i in range(1, 5):
            if line.startswith(f"CAREER_{i}:"):
                parts = line.replace(f"CAREER_{i}:", "").split("|")
                career_entry = {"rank": i}
                for part in parts:
                    part = part.strip()
                    if part.startswith("SCORE:"):
                        try:
                            career_entry["score"] = int(part.replace("SCORE:", "").strip())
                        except ValueError:
                            career_entry["score"] = 80 - (i * 5)
                    elif part.startswith("FIT:"):
                        career_entry["fit_reason"] = part.replace("FIT:", "").strip()
                    elif part.startswith("SKILLS_NEEDED:"):
                        career_entry["skills_needed"] = [
                            s.strip() for s in part.replace("SKILLS_NEEDED:", "").split(",")
                        ]
                    elif part.startswith("NEXT_STEP:"):
                        career_entry["next_step"] = part.replace("NEXT_STEP:", "").strip()
                    else:
                        career_entry.setdefault("title", part)
                careers.append(career_entry)

        if line.startswith("SKILL_GAPS:"):
            skill_gaps = [s.strip() for s in line.replace("SKILL_GAPS:", "").split(",")]
        elif line.startswith("NEXT_STEPS:"):
            next_steps = [s.strip() for s in line.replace("NEXT_STEPS:", "").split(";")]
        elif line.startswith("SUMMARY:"):
            summary = line.replace("SUMMARY:", "").strip()

    # Fallback values if parsing yields incomplete data
    if not careers:
        careers = [
            {
                "rank": 1, "title": "Data Scientist", "score": 92,
                "fit_reason": "Excellent match for your STEM strengths and analytical mindset. High market demand aligns with your profile.",
                "skills_needed": ["Python", "Machine Learning", "SQL"],
                "next_step": "Complete an applied ML course (Coursera / fast.ai)",
            },
            {
                "rank": 2, "title": "Software Engineer (AI/ML)", "score": 87,
                "fit_reason": "Your mathematics strength and tech interest make this a natural fit. Growing field with premium compensation.",
                "skills_needed": ["Deep Learning", "Cloud Platforms", "Git"],
                "next_step": "Build a personal project using TensorFlow or PyTorch",
            },
            {
                "rank": 3, "title": "Cybersecurity Analyst", "score": 78,
                "fit_reason": "Problem-solving aptitude suits threat analysis. Critical global shortage means strong job security.",
                "skills_needed": ["Network Security", "Python Scripting", "CompTIA Security+"],
                "next_step": "Earn CompTIA Security+ certification",
            },
            {
                "rank": 4, "title": "Cloud Solutions Architect", "score": 74,
                "fit_reason": "Analytical background translates well to infrastructure design. High salary ceiling in this role.",
                "skills_needed": ["AWS/Azure/GCP", "System Design", "Terraform"],
                "next_step": "Complete AWS Cloud Practitioner certification",
            },
        ]
    if not skill_gaps:
        skill_gaps = ["Cloud Infrastructure", "Communication Skills", "Version Control (Git)", "Statistical Modeling", "DevOps Practices"]
    if not next_steps:
        next_steps = [
            "Enroll in 'Machine Learning Specialization' on Coursera (Andrew Ng)",
            "Obtain AWS Cloud Practitioner or Google Associate Cloud Engineer certification",
            "Build a portfolio project on GitHub demonstrating data analysis or ML",
        ]
    if not summary:
        summary = (
            f"Based on a comprehensive analysis of your academic strengths, personal interests, and current "
            f"labor market dynamics, {student_name} is well-positioned for a rewarding career in the "
            f"technology sector. Your STEM foundation, combined with strong analytical aptitude, aligns "
            f"perfectly with high-growth fields like data science and AI engineering. Focus on closing "
            f"the identified skill gaps through targeted learning, and you'll be competitive in the job "
            f"market within 12–18 months."
        )

    return {
        "careers": careers,
        "skill_gaps": skill_gaps,
        "next_steps": next_steps,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Flask Application
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")


@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__) or ".", "index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Main orchestration endpoint – runs all four agents sequentially and returns
    the combined recommendation payload.

    Expected JSON body:
    {
      "student_name": "...",
      "grade_level": "...",
      "gpa": "...",
      "subjects": [{"name": "...", "marks": 85, "max_marks": 100}, ...],
      "interests": ["..."],
      "skills": ["..."],
      "personality_traits": ["..."],
      "hobbies": "...",
      "work_style": "..."
    }
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        student_name = data.get("student_name", "Student")

        # --- Agent 1: Academic Analysis ---
        logger.info("Agent 1: Academic Analysis started")
        academic_result = academic_analysis_agent({
            "subjects": data.get("subjects", []),
            "grade_level": data.get("grade_level", ""),
            "gpa": data.get("gpa", ""),
        })
        logger.info("Agent 1: Complete – strengths: %s", academic_result.get("strengths"))

        # --- Agent 2: Interest & Aptitude ---
        logger.info("Agent 2: Interest & Aptitude started")
        aptitude_result = interest_aptitude_agent({
            "interests": data.get("interests", []),
            "skills": data.get("skills", []),
            "personality_traits": data.get("personality_traits", []),
            "hobbies": data.get("hobbies", ""),
            "work_style": data.get("work_style", ""),
        })
        logger.info("Agent 2: Complete – aptitude areas: %s", aptitude_result.get("aptitude_areas"))

        # --- Agent 3: Labor Market (RAG) ---
        logger.info("Agent 3: Labor Market RAG started")
        career_context = (
            f"{academic_result.get('overall_profile', '')} "
            f"Interests: {', '.join(aptitude_result.get('interpreted_interests', []))}. "
            f"Aptitudes: {', '.join(aptitude_result.get('aptitude_areas', []))}."
        )
        market_result = labor_market_agent(career_context)
        logger.info("Agent 3: Complete – trends: %s", market_result.get("top_trends"))

        # --- Agent 4: Career Recommendations ---
        logger.info("Agent 4: Career Recommendation started")
        recommendation = career_recommendation_agent(
            academic_result, aptitude_result, market_result, student_name
        )
        logger.info("Agent 4: Complete – %d pathways generated", len(recommendation.get("careers", [])))

        return jsonify({
            "student_name": student_name,
            "academic": academic_result,
            "aptitude": aptitude_result,
            "market": market_result,
            "recommendation": recommendation,
            "model_used": _MODEL_ID if WATSONX_AVAILABLE else "demo-mock",
            "rag_enabled": RAG_AVAILABLE,
            "demo_mode": not WATSONX_AVAILABLE,
        })

    except Exception as exc:
        logger.exception("Unhandled error in /api/analyze")
        return jsonify({"error": f"Analysis failed: {str(exc)}"}), 500


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "watsonx_available": WATSONX_AVAILABLE,
        "rag_available": RAG_AVAILABLE,
        "demo_mode": not WATSONX_AVAILABLE,
        "model": _MODEL_ID if WATSONX_AVAILABLE else "demo-mock",
        "corpus_size": len(CAREER_CORPUS),
    })


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug)
