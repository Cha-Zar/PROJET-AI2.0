"""
Moteur d'inférence par chaînage avant pour le système expert SysDiag.

Le front ne contient plus la logique de sélection des questions ni l'inférence.
Cette logique vit ici, en Python, à partir de trois fichiers JSON :
- data/rules.json
- data/questions.json
- data/interview_flow.json
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
RULES_PATH = BASE_DIR / "data" / "rules.json"
QUESTIONS_PATH = BASE_DIR / "data" / "questions.json"
FLOW_PATH = BASE_DIR / "data" / "interview_flow.json"
DEPENDENCIES_PATH = BASE_DIR / "data" / "dependencies.json"
CONDITIONS_PATH = BASE_DIR / "data" / "conditions.json"


class KnowledgeBase:
    """Charge et valide la base de connaissances."""

    def __init__(
        self,
        rules_path: str | Path | None = None,
        questions_path: str | Path | None = None,
        flow_path: str | Path | None = None,
        dependencies_path: str | Path | None = None,
        conditions_path: str | Path | None = None,
    ):
        with open(rules_path or RULES_PATH, "r", encoding="utf-8") as handle:
            rules_data = json.load(handle)
        with open(questions_path or QUESTIONS_PATH, "r", encoding="utf-8") as handle:
            questions_data = json.load(handle)
        with open(flow_path or FLOW_PATH, "r", encoding="utf-8") as handle:
            flow_data = json.load(handle)
        with open(dependencies_path or DEPENDENCIES_PATH, "r", encoding="utf-8") as handle:
            dependencies_data = json.load(handle)
        with open(conditions_path or CONDITIONS_PATH, "r", encoding="utf-8") as handle:
            conditions_data = json.load(handle)

        self.settings = {
            **rules_data.get("settings", {}),
            **conditions_data.get("settings", {}),
        }
        self.rules = rules_data["rules"]
        questions_list = questions_data.get("symptom_questions") or questions_data.get(
            "questions"
        )
        if not isinstance(questions_list, list):
            raise ValueError(
                "questions.json doit contenir 'symptom_questions' ou 'questions'."
            )
        self.symptom_questions = {item["id"]: item for item in questions_list}
        self.flow = flow_data
        self.dependencies = dependencies_data.get("dependencies", {})
        configured_roots = dependencies_data.get("root_symptoms", [])

        known_questions = set(self.symptom_questions)
        self.screening_order = [
            question_id
            for question_id in flow_data.get("screening_order", [])
            if question_id in known_questions
        ]
        self.priority_order = [
            question_id
            for question_id in flow_data.get("priority_order", [])
            if question_id in known_questions
        ]

        raw_triggers = flow_data.get("question_triggers", {})
        self.question_triggers = {}
        for question_id, triggers in raw_triggers.items():
            if question_id not in known_questions:
                continue
            valid_triggers = [
                trigger
                for trigger in triggers
                if trigger.get("question") in known_questions and "value" in trigger
            ]
            if valid_triggers:
                self.question_triggers[question_id] = valid_triggers

        raw_contradictions = flow_data.get("contradictions", [])
        self.contradictions = []
        for contradiction in raw_contradictions:
            conditions = contradiction.get("conditions", {})
            if not isinstance(conditions, dict):
                continue
            if set(conditions).issubset(known_questions):
                self.contradictions.append(contradiction)

        raw_inconsistencies = conditions_data.get("inconsistency_rules", [])
        self.inconsistency_rules = []
        for inconsistency in raw_inconsistencies:
            conditions = inconsistency.get("if", {})
            if not isinstance(conditions, dict):
                continue
            if set(conditions).issubset(known_questions):
                self.inconsistency_rules.append(inconsistency)
        self.priority_index = {
            question_id: index for index, question_id in enumerate(self.priority_order)
        }

        if isinstance(configured_roots, list) and configured_roots:
            self.root_symptoms = {
                question_id for question_id in configured_roots if question_id in known_questions
            }
        else:
            # Fallback: les symptômes de screening sont traités comme racines.
            self.root_symptoms = set(self.screening_order)

        raw_rule_dependencies = dependencies_data.get("rule_dependencies", {})
        self.rule_roots: dict[str, set[str]] = {}
        for rule in self.rules:
            rule_id = rule["id"]
            dependency_entry = raw_rule_dependencies.get(rule_id, {})
            explicit_roots = dependency_entry.get("root_symptoms", [])
            roots = {
                question_id for question_id in explicit_roots if question_id in self.root_symptoms
            }
            if not roots:
                roots = self._infer_rule_roots_from_conditions(rule.get("conditions", {}))
            self.rule_roots[rule_id] = roots

        self.min_answers = int(self.settings.get("min_answers", 5))
        self.max_diagnoses = int(self.settings.get("max_diagnoses", 5))
        self.max_questions = int(self.settings.get("max_questions", 18))

        self._validate()

    def _validate(self) -> None:
        known_questions = set(self.symptom_questions)

        for rule in self.rules:
            unknown = set(rule["conditions"]) - known_questions
            if unknown:
                raise ValueError(
                    f"Règle {rule['id']} avec symptômes inconnus: {sorted(unknown)}"
                )

        for question_id, triggers in self.question_triggers.items():
            if question_id not in known_questions:
                continue
            for trigger in triggers:
                if trigger["question"] not in known_questions:
                    continue

        for question_id, dependencies in self.dependencies.items():
            if question_id not in known_questions:
                continue
            for dependency in dependencies:
                dependency_question = dependency.get("question")
                if dependency_question not in known_questions:
                    continue

    def get_rule(self, rule_id: str) -> dict | None:
        for rule in self.rules:
            if rule["id"] == rule_id:
                return rule
        return None

    def get_question(self, symptom_id: str) -> dict | None:
        return self.symptom_questions.get(symptom_id)

    @lru_cache(maxsize=None)
    def infer_question_roots(self, question_id: str) -> tuple[str, ...]:
        if question_id in self.root_symptoms:
            return (question_id,)

        roots: set[str] = set()
        for dependency in self.dependencies.get(question_id, []):
            parent_question = dependency.get("question")
            # Pour déterminer la branche logique, on ne remonte que les prérequis positifs.
            if dependency.get("value") is not True:
                continue
            if parent_question in self.symptom_questions:
                roots.update(self.infer_question_roots(parent_question))

        return tuple(sorted(roots))

    def _infer_rule_roots_from_conditions(self, conditions: dict[str, Any]) -> set[str]:
        roots: set[str] = set()
        for symptom_id in conditions:
            roots.update(self.infer_question_roots(symptom_id))
        return roots


class FactBase:
    """Base de faits issue des réponses utilisateur."""

    def __init__(self):
        self.facts: dict[str, bool | None] = {}
        self.unknown: set[str] = set()

    @classmethod
    def from_payload(cls, answers: dict[str, Any] | None) -> "FactBase":
        fact_base = cls()
        for symptom_id, value in (answers or {}).items():
            normalized = cls._normalize_value(value)
            if normalized is True:
                fact_base.assert_fact(symptom_id, True)
            elif normalized is False:
                fact_base.assert_fact(symptom_id, False)
            else:
                fact_base.mark_unknown(symptom_id)
        return fact_base

    @staticmethod
    def _normalize_value(value: Any) -> bool | None:
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"yes", "oui", "true"}:
                return True
            if lowered in {"no", "non", "false"}:
                return False
            if lowered in {"unknown", "inconnu", "?"}:
                return None
        raise ValueError(f"Réponse invalide: {value!r}")

    def assert_fact(self, symptom_id: str, value: bool) -> None:
        self.facts[symptom_id] = value
        self.unknown.discard(symptom_id)

    def mark_unknown(self, symptom_id: str) -> None:
        self.facts[symptom_id] = None
        self.unknown.add(symptom_id)

    def get(self, symptom_id: str) -> bool | None:
        return self.facts.get(symptom_id)

    def count_all_answers(self) -> int:
        return len(self.facts)

    def clear(self) -> None:
        self.facts.clear()
        self.unknown.clear()


class InferenceEngine:
    """Chaînage avant + planification cohérente des questions."""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base

    def evaluate_rule(self, rule: dict, fact_base: FactBase) -> dict:
        conditions = rule["conditions"]
        unknown_conditions: list[str] = []
        matched_conditions: list[str] = []
        contradicted_conditions: list[str] = []
        base_confidence = rule.get("confidence", 0.8)

        for symptom_id, expected_value in conditions.items():
            actual = fact_base.get(symptom_id)
            if actual is None:
                unknown_conditions.append(symptom_id)
            elif actual == expected_value:
                matched_conditions.append(symptom_id)
            else:
                contradicted_conditions.append(symptom_id)

        if contradicted_conditions:
            return {
                "rule": rule,
                "matched": False,
                "confidence": 0.0,
                "matched_conditions": matched_conditions,
                "unknown_conditions": unknown_conditions,
                "contradicted_conditions": contradicted_conditions,
            }

        if not matched_conditions:
            return {
                "rule": rule,
                "matched": False,
                "confidence": 0.0,
                "matched_conditions": [],
                "unknown_conditions": unknown_conditions,
                "contradicted_conditions": [],
            }

        confidence = base_confidence * (1 - 0.1 * len(unknown_conditions))
        return {
            "rule": rule,
            "matched": True,
            "confidence": round(confidence, 2),
            "matched_conditions": matched_conditions,
            "unknown_conditions": unknown_conditions,
            "contradicted_conditions": [],
        }

    def get_candidate_evaluations(self, fact_base: FactBase) -> list[dict]:
        active_roots = self.get_active_roots(fact_base)
        evaluations = []
        for rule in self.kb.rules:
            rule_roots = self.kb.rule_roots.get(rule["id"], set())
            if active_roots:
                if not rule_roots:
                    continue
                if not rule_roots.intersection(active_roots):
                    continue

            evaluation = self.evaluate_rule(rule, fact_base)
            if not evaluation["contradicted_conditions"]:
                evaluations.append(evaluation)
        return evaluations

    def get_active_roots(self, fact_base: FactBase) -> set[str]:
        return {
            root_id
            for root_id in self.kb.root_symptoms
            if fact_base.get(root_id) is True
        }

    def is_question_in_active_branch(self, question_id: str, fact_base: FactBase) -> bool:
        active_roots = self.get_active_roots(fact_base)
        if not active_roots:
            return True

        question_roots = set(self.kb.infer_question_roots(question_id))
        if not question_roots:
            # Une question sans branche explicite ne doit pas échapper au filtrage strict.
            return False
        return bool(question_roots.intersection(active_roots))

    def is_question_triggered(self, question_id: str, fact_base: FactBase) -> bool:
        triggers = self.kb.question_triggers.get(question_id)
        if not triggers:
            return True
        for trigger in triggers:
            trigger_question = trigger["question"]
            if trigger_question not in self.kb.symptom_questions:
                continue
            if fact_base.get(trigger_question) == trigger["value"]:
                return True
        return False

    def are_dependencies_satisfied(self, question_id: str, fact_base: FactBase) -> bool:
        dependencies = self.kb.dependencies.get(question_id)
        if not dependencies:
            return True

        # Dépendances fortes: on ne pose pas la question tant que les prérequis ne sont pas validés.
        for dependency in dependencies:
            dependency_question = dependency["question"]
            if dependency_question not in self.kb.symptom_questions:
                continue
            dependency_value = dependency["value"]
            if fact_base.get(dependency_question) != dependency_value:
                return False

        return True

    def question_priority(self, question_id: str) -> int:
        return self.kb.priority_index.get(question_id, len(self.kb.priority_order) + 100)

    def score_question(self, question_id: str, evaluations: list[dict]) -> float:
        score = 0.0
        score += max(0, 600 - self.question_priority(question_id) * 5)

        for evaluation in evaluations:
            if question_id not in evaluation["rule"]["conditions"]:
                continue
            score += evaluation["rule"].get("confidence", 0.8) * 100
            score += len(evaluation["matched_conditions"]) * 20
            if evaluation["matched_conditions"]:
                score += 35
            if question_id in evaluation["unknown_conditions"]:
                score += 25

        return score

    def get_next_question(self, fact_base: FactBase) -> str | None:
        if self.kb.max_questions > 0 and fact_base.count_all_answers() >= self.kb.max_questions:
            return None

        answered = set(fact_base.facts)
        evaluations = self.get_candidate_evaluations(fact_base)

        follow_up_candidates: set[str] = set()
        for evaluation in evaluations:
            for symptom_id in evaluation["unknown_conditions"]:
                if symptom_id in answered:
                    continue
                if self.is_question_triggered(
                    symptom_id, fact_base
                ) and self.are_dependencies_satisfied(
                    symptom_id, fact_base
                ) and self.is_question_in_active_branch(symptom_id, fact_base):
                    follow_up_candidates.add(symptom_id)

        if follow_up_candidates:
            return max(
                follow_up_candidates,
                key=lambda symptom_id: (
                    self.score_question(symptom_id, evaluations),
                    -self.question_priority(symptom_id),
                    symptom_id,
                ),
            )

        for symptom_id in self.kb.screening_order:
            if symptom_id in answered:
                continue
            if self.is_question_triggered(
                symptom_id, fact_base
            ) and self.are_dependencies_satisfied(
                symptom_id, fact_base
            ) and self.is_question_in_active_branch(symptom_id, fact_base):
                return symptom_id

        remaining_candidates: set[str] = set()
        for evaluation in evaluations:
            for symptom_id in evaluation["rule"]["conditions"]:
                if symptom_id not in answered and self.is_question_triggered(
                    symptom_id, fact_base
                ) and self.are_dependencies_satisfied(
                    symptom_id, fact_base
                ) and self.is_question_in_active_branch(symptom_id, fact_base):
                    remaining_candidates.add(symptom_id)

        if remaining_candidates:
            return max(
                remaining_candidates,
                key=lambda symptom_id: (
                    self.score_question(symptom_id, evaluations),
                    -self.question_priority(symptom_id),
                    symptom_id,
                ),
            )

        return None

    def run(self, fact_base: FactBase, skip_min_check: bool = False) -> list[dict]:
        if not skip_min_check and fact_base.count_all_answers() < self.kb.min_answers:
            raise ValueError(
                f"Pas assez de réponses : {fact_base.count_all_answers()}/{self.kb.min_answers}. "
                f"Répondez à au moins {self.kb.min_answers} questions."
            )

        diagnoses = []
        for rule in self.kb.rules:
            evaluation = self.evaluate_rule(rule, fact_base)
            if evaluation["matched"]:
                diagnoses.append(
                    {
                        "rule_id": rule["id"],
                        "category": rule["category"],
                        "conclusion": rule["conclusion"],
                        "solution": rule["solution"],
                        "confidence": evaluation["confidence"],
                        "conditions": rule["conditions"],
                        "conditions_used": list(rule["conditions"].keys()),
                    }
                )

        diagnoses.sort(key=lambda item: item["confidence"], reverse=True)
        return diagnoses[: self.kb.max_diagnoses]

    def explain(self, diagnosis: dict, fact_base: FactBase) -> str:
        lines = [
            f"Règle déclenchée : {diagnosis['rule_id']}",
            f"Catégorie : {diagnosis['category']}",
            "",
            "Raisonnement :",
        ]

        rule = self.kb.get_rule(diagnosis["rule_id"])
        if rule:
            for symptom_id, expected in rule["conditions"].items():
                question = self.kb.get_question(symptom_id)
                question_text = question["question"] if question else symptom_id
                actual = fact_base.get(symptom_id)
                if actual is None:
                    status = "?"
                elif actual == expected:
                    status = "OK"
                else:
                    status = "NON"
                lines.append(
                    f" - {status} {question_text} (attendu: {'oui' if expected else 'non'})"
                )

        lines.extend(
            [
                "",
                f"Conclusion : {diagnosis['conclusion']}",
                f"Confiance : {int(diagnosis['confidence'] * 100)}%",
                f"Solution : {diagnosis['solution']}",
            ]
        )
        return "\n".join(lines)

    def detect_inconsistencies(self, fact_base: FactBase) -> list[str]:
        messages: list[str] = []

        for inconsistency in self.kb.inconsistency_rules:
            matches = True
            for symptom_id, expected in inconsistency.get("if", {}).items():
                if fact_base.get(symptom_id) != expected:
                    matches = False
                    break
            if matches:
                messages.append(inconsistency.get("message", "Incohérence détectée."))

        for contradiction in self.kb.contradictions:
            matches = True
            for symptom_id, expected in contradiction["conditions"].items():
                if fact_base.get(symptom_id) != expected:
                    matches = False
                    break
            if matches:
                messages.append(contradiction["message"])
        return messages

    def build_interview_state(self, fact_base: FactBase) -> dict:
        reached_max_questions = (
            self.kb.max_questions > 0
            and fact_base.count_all_answers() >= self.kb.max_questions
        )
        next_question_id = None if reached_max_questions else self.get_next_question(fact_base)
        next_question = None
        if next_question_id:
            next_question = self.kb.get_question(next_question_id)

        return {
            "answers_count": fact_base.count_all_answers(),
            "min_answers": self.kb.min_answers,
            "max_diagnoses": self.kb.max_diagnoses,
            "max_questions": self.kb.max_questions,
            "candidate_rules": len(self.get_candidate_evaluations(fact_base)),
            "next_question": next_question,
            "inconsistencies": self.detect_inconsistencies(fact_base),
            "can_diagnose": (
                fact_base.count_all_answers() >= self.kb.min_answers
                or next_question is None
                or reached_max_questions
            ),
        }


def build_engine() -> InferenceEngine:
    return InferenceEngine(KnowledgeBase())


def run_cli_session() -> None:
    print("\n" + "=" * 60)
    print("  SYSTÈME EXPERT · DIAGNOSTIC DE PANNE INFORMATIQUE")
    print(f"  Minimum {KnowledgeBase().min_answers} réponses")
    print("=" * 60 + "\n")

    kb = KnowledgeBase()
    engine = InferenceEngine(kb)
    fact_base = FactBase()

    while True:
        next_question_id = engine.get_next_question(fact_base)
        if not next_question_id:
            break

        question = kb.get_question(next_question_id)
        if not question:
            break

        answer = input(f"{question['question']} (O/N/?) : ").strip().upper()
        if answer == "O":
            fact_base.assert_fact(next_question_id, True)
        elif answer == "N":
            fact_base.assert_fact(next_question_id, False)
        else:
            fact_base.mark_unknown(next_question_id)

    print("\nRésultats :\n")
    try:
        diagnoses = engine.run(fact_base, skip_min_check=True)
    except ValueError as exc:
        print(exc)
        return

    if not diagnoses:
        print("Aucun diagnostic clair.")
        return

    for index, diagnosis in enumerate(diagnoses, 1):
        print(f"{index}. {diagnosis['conclusion']} ({int(diagnosis['confidence'] * 100)}%)")


if __name__ == "__main__":
    run_cli_session()