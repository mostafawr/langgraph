import networkx as nx

def create_skills_graph():
    """
    Creates a comprehensive directed graph of technical skills.
    """
    G = nx.DiGraph()

    # --- THE BIG STATIC LIST ---
    edges = [
        # --- Software Engineering Foundations ---
        ("Computer Science", "Algorithms"),
        ("Computer Science", "Data Structures"),
        ("Computer Science", "Design Patterns"),
        ("Design Patterns", "MVC"),
        ("Design Patterns", "Singleton"),
        ("Design Patterns", "Microservices"),
        ("Computer Science", "Git"),

        # --- Programming Languages ---
        ("Programming", "Python"),
        ("Programming", "JavaScript"),
        ("Programming", "Java"),
        ("Programming", "C++"),
        ("Programming", "Go"),
        ("Programming", "Rust"),
        ("Programming", "TypeScript"),

        # --- Web Development (Frontend) ---
        ("Web Development", "HTML"),
        ("Web Development", "CSS"),
        ("CSS", "Sass"),
        ("CSS", "Tailwind"),
        ("Web Development", "JavaScript"),
        ("JavaScript", "TypeScript"),
        ("JavaScript", "React"),
        ("React", "Next.js"),
        ("React", "Redux"),
        ("JavaScript", "Vue.js"),
        ("Vue.js", "Nuxt.js"),
        ("JavaScript", "Angular"),
        ("Web Development", "Web Accessibility"),

        # --- Web Development (Backend) ---
        ("Backend Development", "API Design"),
        ("API Design", "REST"),
        ("API Design", "GraphQL"),
        ("Backend Development", "Python"),
        ("Python", "Django"),
        ("Python", "FastAPI"),
        ("Python", "Flask"),
        ("Backend Development", "Node.js"),
        ("Node.js", "Express.js"),
        ("Node.js", "NestJS"),
        ("Backend Development", "Java"),
        ("Java", "Spring Boot"),
        ("Backend Development", "Go"),
        ("Go", "Gin"),

        # --- Mobile Development ---
        ("Mobile Development", "iOS"),
        ("iOS", "Swift"),
        ("iOS", "SwiftUI"),
        ("Mobile Development", "Android"),
        ("Android", "Kotlin"),
        ("Android", "Jetpack Compose"),
        ("Mobile Development", "Cross-Platform"),
        ("Cross-Platform", "React Native"),
        ("Cross-Platform", "Flutter"),

        # --- Data Science & AI ---
        ("Data Science", "Python"),
        ("Data Science", "Statistics"),
        ("Python", "NumPy"),
        ("Python", "Pandas"),
        ("Data Science", "Machine Learning"),
        ("Machine Learning", "Scikit-Learn"),
        ("Machine Learning", "Deep Learning"),
        ("Deep Learning", "TensorFlow"),
        ("Deep Learning", "PyTorch"),
        ("Deep Learning", "Keras"),
        ("Data Science", "Data Visualization"),
        ("Data Visualization", "Matplotlib"),
        ("Data Visualization", "Seaborn"),
        ("Data Visualization", "Tableau"),
        ("Machine Learning", "NLP"),
        ("NLP", "HuggingFace"),
        ("NLP", "OpenAI API"),

        # --- Databases ---
        ("Databases", "SQL"),
        ("SQL", "PostgreSQL"),
        ("SQL", "MySQL"),
        ("SQL", "SQLite"),
        ("Databases", "NoSQL"),
        ("NoSQL", "MongoDB"),
        ("NoSQL", "Redis"),
        ("NoSQL", "Cassandra"),
        ("Databases", "Graph DB"),
        ("Graph DB", "Neo4j"),

        # --- DevOps & Cloud ---
        ("DevOps", "CI/CD"),
        ("CI/CD", "Jenkins"),
        ("CI/CD", "GitHub Actions"),
        ("CI/CD", "GitLab CI"),
        ("DevOps", "Containerization"),
        ("Containerization", "Docker"),
        ("Containerization", "Kubernetes"),
        ("DevOps", "Infrastructure as Code"),
        ("Infrastructure as Code", "Terraform"),
        ("Infrastructure as Code", "Ansible"),
        ("Cloud Computing", "AWS"),
        ("AWS", "EC2"),
        ("AWS", "S3"),
        ("AWS", "Lambda"),
        ("Cloud Computing", "Azure"),
        ("Cloud Computing", "Google Cloud"),

        # --- Cybersecurity ---
        ("Cybersecurity", "Network Security"),
        ("Cybersecurity", "Penetration Testing"),
        ("Penetration Testing", "Kali Linux"),
        ("Penetration Testing", "Metasploit"),
        ("Cybersecurity", "App Security"),
        ("App Security", "OWASP Top 10"),
        ("Cybersecurity", "Cryptography"),
    ]

    # Add edges to graph (normalizing to lowercase for consistency)
    for parent, child in edges:
        G.add_edge(parent.lower(), child.lower())

    return G

def get_skill_relatives(graph, skill):
    """
    Gets the parent and children of a skill in the graph.
    """
    skill = skill.lower()
    
    if skill not in graph:
        return None, None

    parents = list(graph.predecessors(skill))
    children = list(graph.successors(skill))
    return parents, children

if __name__ == '__main__':
    skills_graph = create_skills_graph()
    
    # Test 1: Python
    skill = "Python"
    parents, children = get_skill_relatives(skills_graph, skill)
    print(f"--- {skill} ---")
    print(f"Parents (Super-skills): {parents}")
    print(f"Children (Sub-skills):  {children}\n")

    # Test 2: React
    skill = "React"
    parents, children = get_skill_relatives(skills_graph, skill)
    print(f"--- {skill} ---")
    print(f"Parents (Super-skills): {parents}")
    print(f"Children (Sub-skills):  {children}\n")
    
    # Test 3: DevOps
    skill = "DevOps"
    parents, children = get_skill_relatives(skills_graph, skill)
    print(f"--- {skill} ---")
    print(f"Parents (Super-skills): {parents}")
    print(f"Children (Sub-skills):  {children}\n")
    
    print(f"Total Skills in Graph: {skills_graph.number_of_nodes()}")
    print(f"Total Connections: {skills_graph.number_of_edges()}")