from __future__ import annotations

import sys
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory


BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference_engine import FactBase, InferenceEngine, KnowledgeBase  # noqa: E402


app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")

ENGINE_LOCK = threading.Lock()
KB = KnowledgeBase()
ENGINE = InferenceEngine(KB)
SESSIONS: dict[str, FactBase] = {}


def _current_engine() -> InferenceEngine:
    with ENGINE_LOCK:
        return ENGINE


def _serialize_fact_base(fact_base: FactBase) -> dict[str, bool | None]:
    return dict(fact_base.facts)


def _normalize_answer_value(value: str) -> bool | None:
    if value == "yes":
        return True
    if value == "no":
        return False
    if value == "unknown":
        return None
    raise ValueError("Réponse invalide. Utilisez 'yes', 'no' ou 'unknown'.")


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "interface_SysDiag.html")


@app.get("/api/bootstrap")
def bootstrap():
    engine = _current_engine()
    kb = engine.kb
    return jsonify(
        {
            "settings": {
                "min_answers": kb.min_answers,
                "max_diagnoses": kb.max_diagnoses,
                "max_questions": kb.max_questions,
            },
            "rules_count": len(kb.rules),
            "questions_count": len(kb.symptom_questions),
            "screening_count": len(kb.screening_order),
            "rules": kb.rules,
            "questions": list(kb.symptom_questions.values()),
            "symptom_questions": list(kb.symptom_questions.values()),
        }
    )


@app.post("/api/session/start")
def start_session():
    session_id = str(uuid.uuid4())
    fact_base = FactBase()
    SESSIONS[session_id] = fact_base
    state = _current_engine().build_interview_state(fact_base)
    return jsonify(
        {
            "session_id": session_id,
            "answered_count": state["answers_count"],
            "candidate_rules": state["candidate_rules"],
            "next_question": state["next_question"],
            "can_diagnose": state["can_diagnose"],
            "inconsistencies": state["inconsistencies"],
        }
    )


@app.post("/api/session/<session_id>/answer")
def answer_question(session_id: str):
    fact_base = SESSIONS.get(session_id)
    if not fact_base:
        return jsonify({"error": "Session introuvable."}), 404

    payload = request.get_json(silent=True) or {}
    question_id = payload.get("question_id")
    answer = payload.get("answer")

    if not isinstance(question_id, str) or not question_id:
        return jsonify({"error": "question_id est requis."}), 400

    engine = _current_engine()
    if not engine.kb.get_question(question_id):
        return jsonify({"error": f"Question inconnue: {question_id}"}), 400

    try:
        normalized = _normalize_answer_value(answer)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if normalized is None:
        fact_base.mark_unknown(question_id)
    else:
        fact_base.assert_fact(question_id, normalized)

    state = engine.build_interview_state(fact_base)
    response = {
        "facts": _serialize_fact_base(fact_base),
        "answered_count": state["answers_count"],
        "candidate_rules": state["candidate_rules"],
        "next_question": state["next_question"],
        "can_diagnose": state["can_diagnose"],
        "inconsistencies": state["inconsistencies"],
    }

    if state["next_question"] is None and state["can_diagnose"]:
        response["diagnoses"] = engine.run(fact_base, skip_min_check=True)

    return jsonify(response)


@app.post("/api/session/<session_id>/diagnose")
def session_diagnose(session_id: str):
    fact_base = SESSIONS.get(session_id)
    if not fact_base:
        return jsonify({"error": "Session introuvable."}), 404

    engine = _current_engine()
    try:
        diagnoses = engine.run(fact_base)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "diagnoses": diagnoses,
            "answers_count": fact_base.count_all_answers(),
            "inconsistencies": engine.detect_inconsistencies(fact_base),
            "facts": _serialize_fact_base(fact_base),
        }
    )


@app.post("/api/interview")
def interview_compat():
    payload = request.get_json(silent=True) or {}
    answers = payload.get("answers", {})
    if not isinstance(answers, dict):
        return jsonify({"error": "Le champ 'answers' doit être un objet JSON."}), 400

    fact_base = FactBase.from_payload(answers)
    state = _current_engine().build_interview_state(fact_base)
    return jsonify(state)


@app.post("/api/diagnose")
def diagnose_compat():
    payload = request.get_json(silent=True) or {}
    answers = payload.get("answers", {})
    skip_min_check = bool(payload.get("skip_min_check", False))

    if not isinstance(answers, dict):
        return jsonify({"error": "Le champ 'answers' doit être un objet JSON."}), 400

    fact_base = FactBase.from_payload(answers)
    engine = _current_engine()
    try:
        diagnoses = engine.run(fact_base, skip_min_check=skip_min_check)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "diagnoses": diagnoses,
            "answers_count": fact_base.count_all_answers(),
            "inconsistencies": engine.detect_inconsistencies(fact_base),
        }
    )


@app.post("/api/rules/add")
def add_rule():
    payload = request.get_json(silent=True) or {}
    required_fields = {"id", "category", "conditions", "conclusion", "solution", "confidence"}
    missing = sorted(required_fields - set(payload))
    if missing:
        return jsonify({"error": f"Champs manquants: {', '.join(missing)}"}), 400

    if not isinstance(payload.get("conditions"), dict):
        return jsonify({"error": "Le champ 'conditions' doit être un objet JSON."}), 400

    with ENGINE_LOCK:
        existing_ids = {rule["id"] for rule in KB.rules}
        if payload["id"] in existing_ids:
            return jsonify({"error": f"La règle {payload['id']} existe déjà."}), 400

        unknown_conditions = sorted(set(payload["conditions"]) - set(KB.symptom_questions))
        if unknown_conditions:
            return jsonify(
                {
                    "error": "Conditions inconnues: " + ", ".join(unknown_conditions),
                }
            ), 400

        KB.rules.append(payload)

    return jsonify({"ok": True, "rules_count": len(KB.rules)})


@app.post("/api/rules/delete")
def delete_rule():
    payload = request.get_json(silent=True) or {}
    rule_id = payload.get("id")
    if not isinstance(rule_id, str) or not rule_id:
        return jsonify({"error": "Le champ 'id' est requis."}), 400

    with ENGINE_LOCK:
        initial_count = len(KB.rules)
        KB.rules = [rule for rule in KB.rules if rule.get("id") != rule_id]

        if len(KB.rules) == initial_count:
            return jsonify({"error": f"Règle introuvable: {rule_id}"}), 404

    return jsonify({"ok": True, "rules_count": len(KB.rules)})


if __name__ == "__main__":
    app.run(debug=True)