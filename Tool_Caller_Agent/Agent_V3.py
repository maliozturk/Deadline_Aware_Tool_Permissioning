# =============================================================================
#  DEADLINE-AWARE TOOL PERMISSIONING (DATP)
#  Product Signature: DATP
# ------------------------------------------------------------------------------
#  File: Tool_Caller_Agent/Agent_V3.py
#  Purpose: Run prompt batches against the model and log trace results.
#  Author: Muhammet Ali Ozturk
#  Generated: 2026-01-18
#  Environment: Python 3.9.13
# =============================================================================

import ollama
import time
import random
import sqlite3
import os
from tqdm import tqdm

                               
          
                               
MODEL_NAME = "llama3.1"
DB_NAME = "trace_results_counterfactual.db"

RUN_REPEATS = 10                                        
SLEEP_BETWEEN_CALLS_SEC = 0.0                                            

                               
         
                               
simple_prompts_en = [
"What is the capital of Sweden?",
    "How many teeth do children typically have?",
    "What is the opposite of 'fast'?",
    "Which planet is known for its large red spot?",
    "How many edges does a cube have?",
    "What is the capital of Greece?",
    "What do you call a person who writes books?",
    "How many millimeters are in a centimeter?",
    "What is the color of snow?",
    "Which animal is known for its shell?",
    "What is the capital of India?",
    "How many hours are there in a weekend?",
    "What is the chemical symbol for iron?",
    "Which sense do humans use to hear?",
    "What do you call a baby cat?",
    "How many corners does a rectangle have?",
    "What is the capital of Portugal?",
    "What is the opposite of 'up'?",
    "How many days are there in June?",
    "What is the primary ingredient in cheese?",
    "Which planet is famous for its blue color?",
    "How many grams are in a kilogram?",
    "What is the capital of New Zealand?",
    "What do you call an animal that eats both plants and meat?",
    "How many letters are in the word 'computer'?",
    "What is the color of an emerald?",
    "Which tool is used to cut paper?",
    "How many sides does an octagon have?",
    "What is the capital of Thailand?",
    "What do you call water in gas form?",
    "How many eyes does a human usually have?",
    "What is the opposite of 'early'?",
    "Which animal is known for hopping?",
    "What is the capital of Finland?",
    "How many days are there in a fortnight?",
    "What is the main ingredient in soup?",
    "What is the color of copper?",
    "How many bones are in a human hand?",
    "What do you call a place where books are borrowed?",
    "What is the capital of Chile?",
    "How many sides does a circle have?",
    "Which fruit is typically red or green and grows on trees?",
    "What is the opposite of 'happy'?",
    "How many minutes are in half an hour?",
    "What is the capital of Ireland?",
    "What do you call a device that tells time?",
    "How many arms does an octopus have?",
    "What is the color of the sky on a clear day?",
    "Which season comes after spring?",
"What is the capital of Germany?",
    "How many sides does a triangle have?",
    "What is the freezing point of water in Fahrenheit?",
    "Which planet is known for its rings?",
    "How many continents are there?",
    "What color do you get when you mix blue and red?",
    "What is the largest land animal?",
    "How many hours are in a week?",
    "What gas do plants absorb from the air?",
    "What is the square root of 16?",
    "Which animal is known for its long neck?",
    "What is the capital of South Korea?",
    "How many vowels are in the English alphabet?",
    "What is the main language spoken in Brazil?",
    "What does HTML stand for?",
    "What is the boiling point of water in Fahrenheit?",
    "How many days are in September?",
    "What is the fastest land animal?",
    "What shape has three sides?",
    "What is the currency of Japan?",
    "How many planets orbit the Sun?",
    "What is the chemical symbol for gold?",
    "Which ocean is the largest by surface area?",
    "How many fingers does a human hand have?",
    "What is the opposite of 'cold'?",
    "What do cows produce that humans drink?",
    "What is the tallest animal in the world?",
    "How many letters are in the word 'alphabet'?",
    "What is the capital of Egypt?",
    "What do you call an animal that eats meat?",
    "How many minutes are in two hours?",
    "What is the primary source of energy for Earth?",
    "Which planet is known as Earth's twin?",
    "What is the color of grass?",
    "How many legs does an insect have?",
    "What is the capital of Argentina?",
    "What does CPU stand for?",
    "What do you call frozen rain?",
    "How many sides does a hexagon have?",
    "Which animal is known for changing its color?",
    "What is the capital of Norway?",
    "How many months are in a year?",
    "What is the basic unit of life?",
    "What is the color of coal?",
    "How many wheels does a car usually have?",
    "What is the capital of Mexico?",
    "What do you call a baby dog?",
    "What planet do humans live on?",
    "How many seconds are in an hour?",
    "What is the capital of Turkey?",
    "At what temperature does water boil?",
    "Which company is Elon Musk the CEO of?",
    "What is the code to add an item to a list in Python?",
    "What is 2 times 2?",
    "Which continents is Istanbul located in?",
    "Is the Moon Earth's satellite?",
    "What color do you get when you mix red and yellow?",
    "How many days are there in a year?",
    "How many players play football (soccer) on a team?",
    "What is the largest ocean in the world?",
    "How many hearts does the human body have?",
    "Paris is the capital of which country?",
    "What is the sum of the interior angles of a triangle?",
    "Is the Sun a star?",
    "How do you create a link in HTML?",
    "Which continent is Turkey in?",
    "How many days are there in a week?",
    "Which is denser: salt water or fresh water?",
    "Is a cat a mammal?",
    "How many moons does Earth have?",
    "What are materials that conduct electricity called?",
    "How many centimeters are in a meter?",
    "What does 'hello' mean in English?",
    "What does water turn into when it freezes?",
    "What is the command to print to the console in JavaScript?",
    "What is the smallest prime number?",
    "How many chromosomes do humans have?",
    "What is the currency of Turkey?",
    "How many minutes are in an hour?",
    "What is the chemical symbol for oxygen?",
    "How many sides does a square have?",
    "What is the freezing point of water in Celsius?",
    "Which planet is known as the Red Planet?",
    "How many letters are there in the English alphabet?",
    "What do bees produce?",
    "What is the opposite of 'hot'?",
    "How many continents are there on Earth?",
    "What is the main language spoken in Spain?",
    "What is the name of the device used to measure temperature?",
    "What is the capital of Japan?",
    "How many planets are in the Solar System?",
    "What gas do humans breathe in to survive?",
    "What is 15 minus 7?",
    "Which animal is known as the 'king of the jungle'?",
    "How many hours are there in a day?",
    "What is the color of a ripe banana?",
    "What do you call a shape with five sides?",
    "What is the first month of the year?",
    "How many days are there in February in a non-leap year?",
    "What is the capital of Canada?",
    "How many legs does a spider have?",
    "What is the largest planet in our Solar System?",
    "What is 9 divided by 3?",
    "Which organ pumps blood through the body?",
    "What is the primary color you get by mixing blue and yellow?",
    "How many months have 31 days?",
    "What is the name of the ship that sank in 1912 after hitting an iceberg?",
    "What is the currency of the United Kingdom?",
    "What is the nearest star to Earth?",
    "What is the capital of Australia?",
    "How many degrees are in a full circle?",
    "Which vitamin is produced when the skin is exposed to sunlight?",
    "What is the largest continent on Earth?",
    "What is 100 divided by 4?",
    "What do you call an animal that eats only plants?",
    "How many zeros are in one million?",
    "What is the process by which plants make food using sunlight called?",
    "Which instrument is used to measure air pressure?",
    "How many strings does a standard guitar have?",
"What is the capital of Brazil?",
    "How many bones are in the adult human body?",
    "What is 7 times 8?",
    "Which planet is closest to the Sun?",
    "What is the tallest mountain in the world called?",
    "How many colors are there in a rainbow?",
    "What is the largest mammal on Earth?",
    "What is the name of the force that pulls objects toward Earth?",
    "How many days are there in a leap year?",
    "Which metal is liquid at room temperature?",
"What is the capital of Italy?",
    "How many wheels does a standard bicycle have?",
    "What is 12 plus 19?",
    "Which ocean is on the west coast of the United States?",
    "What is the main ingredient in bread?",
    "What is the largest internal organ in the human body?",
    "How many days are there in the month of April?",
    "What is the name of the line that separates day and night on Earth?",
    "Which animal is famous for black-and-white stripes?",
    "What is the name of the hard outer layer of a tree?",
    "What is the capital of Italy?",
    "How many seconds are in one minute?",
    "What is the largest organ in the human body?",
    "What is 12 plus 19?",
    "Which ocean is on the east coast of the United States?",
    "What do you call frozen water?",
    "How many teeth does an adult human typically have?",
    "Which planet has the most prominent rings?",
    "What is the main ingredient in bread that makes it rise?",
    "What is the name of the tool used to see very small objects in science?"
]



complex_prompts_en = [
    "Explain how incentives can unintentionally encourage harmful behavior in organizations.",
    "Design a scalable logging system and explain how you would index and query logs efficiently.",
    "Analyze how loss aversion influences consumer purchasing decisions.",
    "Explain the difference between synchronous and asynchronous communication in distributed systems.",
    "Propose a strategy for evaluating long-term user satisfaction beyond short-term engagement metrics.",
    "Discuss how path dependence affects economic and technological evolution.",
    "Explain how rate limiting protects APIs from abuse and failure cascades.",
    "Design a framework to balance exploration vs exploitation in recommendation systems.",
    "Analyze how social media algorithms amplify extreme content and propose mitigations.",
    "Explain the difference between precision, recall, and F1-score with concrete examples.",
    "Propose an approach to detect concept drift without labeled data.",
    "Discuss how organizational structure influences software architecture (Conway’s Law).",
    "Explain how auction mechanisms differ (first-price, second-price, VCG) and their incentives.",
    "Design a pricing strategy for a freemium SaaS product using behavioral insights.",
    "Analyze the trade-offs between transparency and security in system design.",
    "Explain how memory hierarchy affects program performance and optimization choices.",
    "Propose a methodology for stress-testing a large-scale distributed system.",
    "Discuss the limits of rational-agent models in economics and AI.",
    "Explain how shadow IT emerges and how organizations can manage it.",
    "Design a human-in-the-loop system for moderating online content at scale.",
    "Analyze how principal–agent problems arise in corporate governance.",
    "Explain the difference between online learning and batch learning in ML systems.",
    "Propose a framework for evaluating the societal impact of new technologies.",
    "Discuss how latency and throughput trade off in system performance engineering.",
    "Explain how adversarial examples challenge machine learning robustness.",
    "Design a rollback strategy for data schema changes in production systems.",
    "Analyze how scarcity framing affects perceived value in marketing.",
    "Explain how distributed locks work and where they fail in practice.",
    "Propose a strategy to align product metrics with long-term company goals.",
    "Discuss the implications of Goodhart’s Law in performance measurement.",
    "Explain how Bayesian updating differs from frequentist inference conceptually.",
    "Design a system to personalize content without creating echo chambers.",
    "Analyze how platform governance choices affect creator incentives.",
    "Explain how eventual consistency can surface user-visible anomalies.",
    "Propose a method to estimate causal lift when randomization is impossible.",
    "Discuss the role of norms and culture in enforcing cooperation.",
    "Explain how cold starts differ for users vs items in recommender systems.",
    "Design an observability strategy covering metrics, logs, and traces.",
    "Analyze the trade-offs between static and dynamic typing in large codebases.",
    "Explain how feedback loops can destabilize complex systems.",
    "Propose a strategy for gradual deprecation of public APIs.",
    "Discuss how interpretability requirements vary across high-stakes domains.",
    "Explain how market power emerges from economies of scale in digital goods.",
    "Design a cross-functional decision-making process for data-driven products.",
    "Analyze how defaults influence user choices and ethical considerations.",
    "Explain how concurrency bugs differ from parallelism bugs.",
    "Propose a framework to evaluate trustworthiness in AI assistants.",
    "Discuss how organizational incentives can distort data reporting.",
    "Explain how resilience engineering differs from traditional risk management.",
    "Design a system to measure true user value rather than proxy metrics.",
"Explain the difference between supervised, unsupervised, and reinforcement learning with real-world examples.",
    "Design a scalable architecture for a real-time chat application and explain key trade-offs.",
    "Analyze how confirmation bias affects decision-making in organizations and propose mitigation strategies.",
    "Explain how blockchain achieves trust without a central authority, including its limitations.",
    "Design an experiment to measure the causal impact of a new feature on user retention.",
    "Discuss how cultural factors influence negotiation strategies in international business.",
    "Explain the difference between horizontal and vertical scaling and when to use each.",
    "Analyze the ethical implications of facial recognition technology in public spaces.",
    "Explain how TCP ensures reliable data transmission over unreliable networks.",
    "Propose a strategy to handle missing data in a large-scale machine learning dataset.",
    "Explain how monetary policy tools influence inflation and unemployment.",
    "Design a fault-tolerant data pipeline for processing streaming events at scale.",
    "Discuss how cognitive load theory impacts instructional design.",
    "Explain the trade-offs between SQL and NoSQL databases for analytical workloads.",
    "Analyze how network effects can create monopolies in digital markets.",
    "Explain how backpressure works in stream processing systems.",
    "Design a user authentication flow that balances security and usability.",
    "Discuss how survivorship bias distorts conclusions in data analysis.",
    "Explain the role of embeddings in semantic search systems.",
    "Propose a framework to evaluate fairness in machine learning models.",
    "Analyze how exchange rates affect international trade balances.",
    "Explain the principles of RESTful API design and common anti-patterns.",
    "Design a monitoring system to detect anomalies in real-time metrics.",
    "Discuss how incentives shape behavior in online platforms.",
    "Explain how container orchestration systems like Kubernetes manage failures.",
    "Analyze the environmental trade-offs of cloud computing versus on-premise infrastructure.",
    "Explain how causal graphs help distinguish correlation from causation.",
    "Design a recommendation system for news while minimizing filter bubbles.",
    "Discuss the psychological factors behind habit formation in product design.",
    "Explain the difference between eventual consistency and linearizability.",
    "Propose a data anonymization strategy that balances privacy and utility.",
    "Analyze how algorithmic decision-making can reinforce social inequalities.",
    "Explain how transformers differ from RNNs in handling long-range dependencies.",
    "Design an access control model for a multi-tenant SaaS platform.",
    "Discuss the limitations of accuracy as a metric in imbalanced classification problems.",
    "Explain how market signaling works in labor markets.",
    "Design a system to detect and mitigate bot activity on a website.",
    "Analyze the trade-offs between centralized and decentralized governance models.",
    "Explain how feature flags enable safer software deployments.",
    "Propose an evaluation framework for explainability in AI systems.",
    "Discuss how time-series forecasting differs from cross-sectional prediction.",
    "Explain how schema evolution can be handled in distributed data systems.",
    "Analyze the role of trust in peer-to-peer marketplaces.",
    "Design a consent management system compliant with data protection laws.",
    "Explain how graph databases differ from relational databases and when to use them.",
    "Discuss the economic rationale behind subscription pricing models.",
    "Explain how model ensembles improve robustness and when they fail.",
    "Design a strategy to prevent data drift in production ML systems.",
    "Analyze how social proof influences user behavior in online reviews.",
    "Explain quantum entanglement as if you were explaining it to a 5-year-old.",
    "Design a database schema for an e-commerce website and explain the tables.",
    "Analyze the economic impacts of climate change on agriculture.",
    "Explain step by step the logic of writing an asynchronous web scraper in Python.",
    "Discuss the ontological basis of the statement 'I think, therefore I am' in philosophy.",
    "What are the logistical and psychological challenges of building a colony on Mars?",
    "Explain how opening theory in chess affects middlegame strategy.",
    "Create an essay outline about the ethical problems of artificial intelligence.",
    "List and elaborate on the legal processes to consider when founding a startup.",
    "Why is the sky blue? Explain in detail with the physical principles.",
    "When designing a recommendation system, how would you solve the cold-start problem? Compare methods.",
    "In modern cryptography, what is 'forward secrecy' and why is it important? Explain with an example scenario.",
    "What are the pros and cons of building an offline-first architecture in a mobile app? Propose a synchronization strategy.",
    "Analyze how inflation affects income distribution through its mechanisms (wages, assets, indebtedness).",
    "What signals would you look at to detect overfitting in a neural network, and how would you intervene?",
    "Discuss the difference between deontological ethics and consequentialism in Kant’s moral philosophy, using an argument structure.",
    "For optimizing public transportation in a city, what data would you collect and which optimization approach would you choose?",
    "Why is distributed tracing necessary in a microservices architecture? Explain with its core components.",
    "Design a process to ensure data quality in a large-scale data pipeline.",
    "Which is more effective for reducing greenhouse gas emissions: a carbon tax or emissions trading? Evaluate depending on conditions.",
    "How would you measure demand elasticity to set a product’s price, and how would you incorporate it into a model?",
    "Explain intuitively and technically how the attention mechanism in transformer-based models affects information flow.",
    "Create an end-to-end compliance plan to make a company’s data GDPR/KVKK-compliant.",
    "In game theory, what does a Nash equilibrium guarantee, and in which cases is it insufficient? Discuss with examples.",
    "Explain the CAP theorem in distributed systems with real-world examples and how it influences architectural decisions.",
    "In A/B testing, how would you detect and prevent p-hacking and multiple-comparisons problems?",
    "How would you study the long-term impact of investment in education on economic growth from a causality perspective?",
    "How would you mitigate the prompt injection threat in an LLM application? Propose a layered defense strategy.",
    "How can you tell if a dataset has selection bias, and which methods would you use to correct it?",
    "When writing a persuasive speech, how would you balance ethos/pathos/logos? Propose a structure.",
"Design an API rate-limiting strategy for a public service: compare token bucket vs leaky bucket and justify your choice.",
    "Explain how gradient descent behaves on a non-convex loss surface, and what techniques help escape saddle points.",
    "Propose an experiment to estimate the causal effect of remote work on productivity, including threats to validity.",
    "Analyze the trade-offs between eventual consistency and strong consistency in a global-scale system with user-facing latency constraints.",
    "Create a threat model for a consumer web app and propose mitigations for the top risks (auth, data leakage, abuse).",
    "Explain the difference between principal component analysis (PCA) and t-SNE/UMAP, and when you would choose each for exploration.",
    "Design a data governance approach for a company adopting a lakehouse architecture, including access control and lineage.",
    "In macroeconomics, explain how an interest rate hike transmits to inflation via multiple channels, and where it can fail.",
    "Draft a policy for responsible AI use in hiring, including fairness evaluation, monitoring, and appeal mechanisms.",
    "Explain how to evaluate an LLM-powered assistant end-to-end (offline metrics, human eval, and online A/B testing) and define success criteria.",
"Design a sharded database strategy for a fast-growing application and explain how you would handle re-sharding with minimal downtime.",
    "Explain how retrieval-augmented generation (RAG) can fail (hallucinations, stale context, retrieval mismatch) and propose concrete mitigations.",
    "Propose a monitoring and alerting plan for a machine learning model in production (data drift, concept drift, performance decay).",
    "Analyze how misinformation spreads on social networks using network effects and incentives, and suggest intervention points.",
    "Compare zero-shot, few-shot, and fine-tuning approaches for adapting an LLM to a domain; include cost, risk, and evaluation considerations.",
    "Explain the economics of two-sided marketplaces and how pricing strategy differs from single-sided markets.",
    "Design an incident response runbook for a suspected credential-stuffing attack, including containment and postmortem steps.",
    "Discuss the philosophical debate between compatibilism and libertarian free will, and how each addresses moral responsibility.",
    "Create a plan to migrate a monolith to microservices while minimizing organizational and technical risk; include sequencing and boundaries.",
    "Explain why correlation does not imply causation using at least three distinct confounding scenarios and how to resolve them empirically.",
    "Explain how differential privacy works at a high level and how you would choose an epsilon value for a real product use-case.",
    "Design a feature store for machine learning: discuss offline vs online serving, consistency, and backfilling.",
    "Compare BERT-style encoder models vs GPT-style decoder models: architectural differences and best-fit tasks.",
    "Propose a methodology to audit a dataset for demographic bias without having explicit demographic labels.",
    "Explain how a compiler optimizes code (e.g., inlining, loop unrolling, dead-code elimination) and the trade-offs involved.",
    "Analyze the impact of supply chain disruptions on inflation and employment, using an aggregate-demand/aggregate-supply lens.",
    "Design an experiment to measure the long-term retention impact of a new onboarding flow, including metric definitions and pitfalls.",
    "Explain the Byzantine Generals Problem and how practical consensus protocols address it in real-world systems.",
    "Outline a secure key management and rotation policy for a cloud-native application handling sensitive user data.",
    "Discuss how narrative framing affects public policy support, referencing cognitive biases and decision-making mechanisms.",
"Design a caching strategy for a high-traffic application: what to cache, eviction policies, and how to prevent cache stampedes.",
    "Explain the bias–variance trade-off and how it guides model selection and regularization choices in practice.",
    "Propose a robust evaluation plan for a credit-scoring model that balances accuracy, fairness, and regulatory constraints.",
    "Analyze the geopolitical and economic consequences of large-scale renewable energy adoption for oil-exporting countries.",
    "Explain how backpropagation works conceptually, and why vanishing/exploding gradients occur in deep networks.",
    "Design a multilingual search system: tokenization, ranking, language detection, and evaluation methodology.",
    "Discuss how institutional incentives can lead to policy lock-in, and propose mechanisms to enable adaptive policymaking.",
    "Compare containerization and virtual machines for production workloads: isolation, performance, and operational trade-offs.",
    "Create a plan for secure software supply chain management (SBOMs, signing, dependency pinning, CI/CD controls).",
    "Explain how to perform a root-cause analysis for a sudden drop in conversion rate, including hypotheses and data checks.",
"Design a fraud-detection system for online payments: features, model choice, latency constraints, and human review loops.",
    "Explain how L1 vs L2 regularization changes model behavior, sparsity, and robustness; include when you’d prefer each.",
    "Propose an architecture for real-time analytics (stream processing): ingestion, storage, windowing, and exactly-once trade-offs.",
    "Analyze how housing supply constraints affect rent inflation and labor mobility, and propose policy interventions with trade-offs.",
    "Explain how public-key infrastructure (PKI) works end-to-end and where trust can break in practice.",
    "Design a privacy-preserving telemetry strategy for a mobile app while keeping it useful for debugging and product decisions.",
    "Discuss the ethics of persuasive design in consumer apps and propose a framework for avoiding dark patterns.",
    "Compare Monte Carlo methods vs deterministic numerical methods for estimating integrals or uncertainties; provide use-cases.",
    "Create a plan to evaluate and reduce carbon footprint of a data center workload, including measurement and optimization levers.",
    "Explain how to build a robust ETL testing strategy (unit, integration, reconciliation) for mission-critical reporting.",
"Design a reliable messaging system for microservices: compare at-least-once vs exactly-once delivery and how you’d implement idempotency.",
    "Explain how feature drift differs from concept drift, and propose detection + remediation steps for each.",
    "Propose a strategy to reduce churn for a subscription product using cohort analysis, segmentation, and experimentation.",
    "Analyze the trade-offs of central bank digital currencies (CBDCs) for privacy, financial stability, and monetary policy.",
    "Explain how to defend an application against SSRF vulnerabilities with concrete controls and validation strategies.",
    "Design a high-quality labeling workflow for training data: guidelines, QA sampling, inter-annotator agreement, and iteration.",
    "Discuss the philosophical problem of induction and how it impacts scientific reasoning and everyday decision-making.",
    "Compare batch processing vs stream processing for anomaly detection: latency, cost, and accuracy implications.",
    "Create an approach to evaluate a ranking/search algorithm using offline metrics (NDCG/MAP) and online metrics; address metric gaming.",
    "Explain how to plan a zero-downtime migration from one authentication system to another (sessions, tokens, rollout, rollback).",
    "Explain how vector databases enable semantic search, and how you would choose an embedding model and distance metric.",
    "Design a robust rollout strategy for a risky backend change (canary, feature flags, rollback) with clear success metrics.",
    "Analyze how algorithmic pricing can lead to tacit collusion, and suggest monitoring or regulatory safeguards.",
    "Propose a secure authentication architecture for a consumer app (MFA, session management, refresh tokens) and common pitfalls.",
    "Compare optimistic vs pessimistic concurrency control in databases, including failure modes and performance trade-offs.",
    "Explain how knowledge distillation works for model compression and where it can degrade performance unexpectedly.",
    "Design a labeling strategy for training a high-quality classifier when labels are expensive (active learning, weak supervision).",
    "Discuss how central bank communication (forward guidance) affects market expectations and economic outcomes, with mechanisms.",
    "Outline an approach to detect and mitigate data leakage in machine learning pipelines, including time-based leakage.",
    "Explain how to construct a decision matrix for build-vs-buy choices in software, including hidden costs and vendor risk."
]




all_prompts = [(p, "simple") for p in simple_prompts_en] + [(p, "complex") for p in complex_prompts_en]
random.shuffle(all_prompts)

                               
                             
                               
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "fast_lookup",
            "description": "Use this for simple, factual, quick questions that do not require reasoning.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The query to answer"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deep_reasoner",
            "description": "Use this for complex, multi-step, philosophical, or coding tasks requiring detailed explanation.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The complex query"}},
                "required": ["query"],
            },
        },
    },
]

                               
         
                               
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS traces_cf (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            run_iter INTEGER,
            prompt_index INTEGER,

            prompt_type TEXT,
            prompt TEXT,

            -- Router phase (optional but useful)
            router_choice TEXT,
            router_latency_sec REAL,
            router_raw_text TEXT,

            -- FAST counterfactual
            fast_total_latency_sec REAL,
            fast_generation_only_sec REAL,
            fast_response_length_char INTEGER,
            fast_response_text TEXT,
            fast_error TEXT,

            -- SLOW counterfactual
            slow_total_latency_sec REAL,
            slow_generation_only_sec REAL,
            slow_response_length_char INTEGER,
            slow_response_text TEXT,
            slow_error TEXT,

            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    return conn

                               
         
                               
def safe_sleep():
    if SLEEP_BETWEEN_CALLS_SEC and SLEEP_BETWEEN_CALLS_SEC > 0:
        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

def timed_chat(model: str, messages, tools=None):
    t0 = time.time()
    try:
        if tools is None:
            resp = ollama.chat(model=model, messages=messages)
        else:
            resp = ollama.chat(model=model, messages=messages, tools=tools)
        elapsed = time.time() - t0
        return resp, elapsed, None
    except Exception as e:
        elapsed = time.time() - t0
        return None, elapsed, str(e)

def extract_router_choice(resp):
    if resp is None:
        return "error"

    msg = resp.get("message", {})
    tool_calls = msg.get("tool_calls", []) or []
    if not tool_calls:
        return "none"

    name = tool_calls[0].get("function", {}).get("name", "")
    if name == "fast_lookup":
        return "fast"
    if name == "deep_reasoner":
        return "slow"
    return f"unknown:{name}"

def build_fast_instruction(prompt: str) -> str:
    return (
        f"Question: {prompt}\n"
        "Instruction: Provide a very short, clear, single-sentence answer. Be concise."
    )

def build_slow_instruction(prompt: str) -> str:
    return (
        f"Question: {prompt}\n"
        "Instruction: Analyze this question in depth. Think step-by-step. Write a detailed and long response."
    )

                               
                 
                               
def run_trace_collection_counterfactual():
    conn = init_db()
    cur = conn.cursor()

    print(f"System starting... Model: {MODEL_NAME}")
    print(f"DB: {os.path.abspath(DB_NAME)}")
    print(f"Total prompts: {len(all_prompts)}")
    print("-" * 60)

    for run_iter in tqdm(range(RUN_REPEATS), desc="Overall Iteration"):
        for idx, (prompt, ptype) in enumerate(all_prompts):
            print(f"[iter={run_iter}] [{idx + 1}/{len(all_prompts)}] {ptype.upper()} prompt: {prompt[:60]}...")

                                  
                                        
                                  
            router_resp, router_elapsed, router_err = timed_chat(
                MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                tools=tools_schema,
            )
            router_choice = extract_router_choice(router_resp) if router_err is None else "error"
            router_raw_text = ""
            if router_resp is not None:
                router_raw_text = (router_resp.get("message", {}).get("content") or "")[:2000]                       

            safe_sleep()

                                  
                                              
                                  
            fast_instruction = build_fast_instruction(prompt)

            fast_total_start = time.time()
            fast_resp, fast_gen_elapsed, fast_err = timed_chat(
                MODEL_NAME,
                messages=[{"role": "user", "content": fast_instruction}],
                tools=None,
            )
            fast_total_elapsed = time.time() - fast_total_start

            fast_text = ""
            fast_len = 0
            if fast_resp is not None:
                fast_text = fast_resp.get("message", {}).get("content") or ""
                fast_len = len(fast_text)

            safe_sleep()

                                  
                                              
                                  
            slow_instruction = build_slow_instruction(prompt)

            slow_total_start = time.time()
            slow_resp, slow_gen_elapsed, slow_err = timed_chat(
                MODEL_NAME,
                messages=[{"role": "user", "content": slow_instruction}],
                tools=None,
            )
            slow_total_elapsed = time.time() - slow_total_start

            slow_text = ""
            slow_len = 0
            if slow_resp is not None:
                slow_text = slow_resp.get("message", {}).get("content") or ""
                slow_len = len(slow_text)

                                  
                               
                                  
            cur.execute(
                """
                INSERT INTO traces_cf (
                    run_iter, prompt_index, prompt_type, prompt,
                    router_choice, router_latency_sec, router_raw_text,
                    fast_total_latency_sec, fast_generation_only_sec, fast_response_length_char, fast_response_text, fast_error,
                    slow_total_latency_sec, slow_generation_only_sec, slow_response_length_char, slow_response_text, slow_error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_iter, idx, ptype, prompt,
                    router_choice, round(router_elapsed, 4), router_raw_text,
                    round(fast_total_elapsed, 4), round(fast_gen_elapsed, 4), fast_len, fast_text, fast_err,
                    round(slow_total_elapsed, 4), round(slow_gen_elapsed, 4), slow_len, slow_text, slow_err,
                ),
            )
            conn.commit()

            print(
                f"   Router={router_choice} ({router_elapsed:.2f}s) | "
                f"FAST={fast_total_elapsed:.2f}s (gen {fast_gen_elapsed:.2f}s) | "
                f"SLOW={slow_total_elapsed:.2f}s (gen {slow_gen_elapsed:.2f}s)"
            )

    print("-" * 60)
    print("Counterfactual trace collection done.")

                       
    cur.execute(
        """
        SELECT prompt_type,
               COUNT(*),
               AVG(fast_generation_only_sec),
               AVG(slow_generation_only_sec)
        FROM traces_cf
        WHERE fast_error IS NULL AND slow_error IS NULL
        GROUP BY prompt_type
        """
    )
    rows = cur.fetchall()
    print("\nSummary (avg generation-only seconds):")
    print(f"{'TYPE':<10} {'N':<8} {'FAST_AVG':<10} {'SLOW_AVG':<10}")
    for r in rows:
        print(f"{r[0]:<10} {r[1]:<8} {r[2]:<10.4f} {r[3]:<10.4f}")

    conn.close()


if __name__ == "__main__":
    run_trace_collection_counterfactual()
