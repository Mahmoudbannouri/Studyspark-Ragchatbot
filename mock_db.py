# mock_db.py
DB = {
    "users": [
        {"id": "u1", "name": "Achref", "role": "student"},
        {"id": "u2", "name": "Selim", "role": "student"},
        {"id": "admin1", "name": "Admin", "role": "admin"},
    ],
    "resources": [
        {"id": "r1", "user_id": "u1", "title": "Biology - DNA Replication",
         "content": "DNA replication involves helicase, primase, DNA polymerase, ligase. Leading and lagging strands. Okazaki fragments..."},
        {"id": "r2", "user_id": "u2", "title": "Physics - Kinematics",
         "content": "Displacement, velocity, acceleration, equations of motion, projectile trajectory..."},
    ],
    "notes": [
        {"id": "n1", "user_id": "u1", "title": "Bio notes wk1",
         "content": "Enzymes in replication: helicase unwinds, primase lays primers, DNA pol extends, ligase seals nicks."},
        {"id": "n2", "user_id": "u2", "title": "Phys prep",
         "content": "Projectile: range depends on initial velocity and angle; air resistance reduces range."},
    ],
    "summaries": [
        {"id": "s1", "user_id": "u1", "source": "r1", "length": "short",
         "content": "DNA replication: helicase, primase, DNA pol, ligase; Okazaki fragments; semi-conservative."},
    ],
    "flashcards": [
        {"id": "f1", "user_id": "u1", "deck": "Bio deck",
         "q": "Enzyme that unwinds DNA?", "a": "Helicase"},
        {"id": "f2", "user_id": "u2", "deck": "Phys deck",
         "q": "Formula for displacement (const accel)?", "a": "s = ut + 1/2 at^2"},
    ],
    "quiz": [
        {"id": "qz1", "user_id": "u1", "subject": "Biology",
         "questions": ["What is helicase?"], "attempts": [{"score": 45, "date": "2025-09-20"}]},
        {"id": "qz2", "user_id": "u2", "subject": "Physics",
         "questions": ["Define acceleration"], "attempts": [{"score": 88, "date": "2025-09-18"}]},
    ],
    "study_plans": [
        {"id": "sp1", "user_id": "u1", "goal": "Bio midterm", "tasks": [
            {"title": "Review DNA replication", "duration_min": 45, "priority": "high"},
            {"title": "Do flashcards", "duration_min": 20, "priority": "med"},
        ]}
    ],
    "qa_sessions": [],
    "jobs": []
}
