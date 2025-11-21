from typing import List, Dict, Tuple, Set
from copy import deepcopy

class Literal:
    def __init__(self, name: str, args: List[str], negated: bool = False):
        self.name = name
        self.args = tuple(args)
        self.negated = negated

    def negate(self) -> 'Literal':
        return Literal(self.name, list(self.args), not self.negated)

    def __eq__(self, other):
        return (self.name == other.name and self.args == other.args and self.negated == other.negated)

    def __hash__(self):
        return hash((self.name, self.args, self.negated))

    def __repr__(self):
        sign = "¬" if self.negated else ""
        if not self.args:
            return f"{sign}{self.name}"
        return f"{sign}{self.name}({', '.join(self.args)})"


def parse_cnf(text: str) -> List[List[Literal]]:
    clauses = []
    for clause_str in text.split(','):
        clause_str = clause_str.strip()
        if not clause_str:
            continue
        literals = []
        parts = clause_str.split(' ∨ ')
        for part in parts:
            part = part.strip()
            negated = part.startswith('¬')
            if negated:
                part = part[1:].strip()
            if '(' in part:
                name, arg_part = part.split('(', 1)
                args = [a.strip() for a in arg_part.rstrip(')').split(',')]
            else:
                name = part
                args = []
            literals.append(Literal(name, args, negated))
        clauses.append(literals)
    return clauses


def apply_subst(lit: Literal, subst: Dict[str, str]) -> Literal:
    new_args = [subst.get(arg, arg) for arg in lit.args]
    return Literal(lit.name, new_args, lit.negated)


def unify(arg1: str, arg2: str, subst: Dict[str, str]) -> bool:
    if arg1 == arg2:
        return True
    if arg1.islower():
        if arg1 in subst and subst[arg1] != arg2:
            return False
        subst[arg1] = arg2
        return True
    if arg2.islower():
        return unify(arg2, arg1, subst)
    return False


def unify_literals(l1: Literal, l2: Literal) -> Dict[str, str] | None:
    if l1.name != l2.name or l1.negated == l2.negated:
        return None
    if len(l1.args) != len(l2.args):
        return None
    subst = {}
    for a1, a2 in zip(l1.args, l2.args):
        if not unify(a1, a2, subst):
            return None
    return subst


class ResolutionEngine:
    def prove(self, cnf_string: str) -> Tuple[bool, List[str]]:
        clauses = parse_cnf(cnf_string)
        return self._resolution_prove(clauses)

    def _resolution_prove(self, initial_clauses: List[List[Literal]]) -> Tuple[bool, List[str]]:
        clauses: List[List[Literal]] = [list(c) for c in initial_clauses]
        steps: List[str] = []
        used: Set[Tuple[str, ...]] = set()
        step_num = 1

        def clause_id(clause: List[Literal]) -> Tuple[str, ...]:
            return tuple(sorted(str(lit) for lit in clause))

        queue = clauses.copy()

        while queue:
            current = queue.pop(0)
            curr_id = clause_id(current)
            if curr_id in used:
                continue
            used.add(curr_id)

            if not current:  
                steps.append(f"Шаг {step_num}: Получен пустой дизъюнкт")
                steps.append("Цель доказана методом резолюций!")
                return True, steps

            for existing in clauses:
                for lit1 in current:
                    for lit2 in existing:
                        if lit1.name == lit2.name and lit1.negated != lit2.negated:
                            subst = unify_literals(lit1, lit2)
                            if subst is not None:
                                # Убираем противоположные литералы
                                new_clause = [l for l in current if l != lit1] + \
                                             [l for l in existing if l != lit2]
                                # Применяем подстановку
                                new_clause = [apply_subst(l, subst) for l in new_clause]
                                # Убираем дубликаты
                                seen = set()
                                new_clause = [l for l in new_clause if l not in seen and not seen.add(l)]

                                expl = f"Шаг {step_num}: {lit1.negate()} + {lit2.negate()} " \
                                       f"→ подстановка {subst} → {' ∨ '.join(map(str, new_clause)) or 'Пустой дизъюнкт'}"
                                steps.append(expl)

                                if not new_clause: 
                                    steps.append("Найдено противоречие → цель доказана!")
                                    return True, steps

                                new_id = clause_id(new_clause)
                                if new_id not in used:
                                    queue.append(new_clause)
                                    clauses.append(new_clause)
                                step_num += 1

        steps.append("Противоречие не найдено — цель НЕ доказана этим набором аксиом.")
        return False, steps


# ======================== ТЕСТЫ ========================
if __name__ == "__main__":
    engine = ResolutionEngine()

    print("=== Тест 1: Сократ ===")
    formulas1 = "Человек(Сократ), ¬Человек(x) ∨ Смертен(x), ¬Смертен(Сократ)"
    proven1, steps1 = engine.prove(formulas1)
    print(f"Формулы: {formulas1}")
    print(f"Доказано: {proven1}")
    print("Шаги доказательства:")
    for i, step in enumerate(steps1, 1):
        print(f"{i}. {step}")
    print()

    print("=== Тест 2: Студент и экзамены ===")
    formulas2 = "Студент(Петя), ¬Сдал(x,МЛиТА) ∨ ¬Сдал(x,Дискретка) ∨ Зачёт(x), Сдал(Петя,Дискретка), ¬Зачёт(Петя)"
    proven2, steps2 = engine.prove(formulas2)
    print(f"Формулы: {formulas2}")
    print(f"Доказано: {proven2}")
    print("Шаги доказательства:")
    for i, step in enumerate(steps2, 1):
        print(f"{i}. {step}")
    print()

    print("=== Тест 3: Все любят Сократа ===")
    formulas3 = "¬Человек(x) ∨ Любит(x,Сократ), Человек(Аристотель), ¬Любит(Аристотель,Сократ)"
    proven3, steps3 = engine.prove(formulas3)
    print(f"Формулы: {formulas3}")
    print(f"Доказано: {proven3}")
    print("Шаги доказательства:")
    for i, step in enumerate(steps3, 1):
        print(f"{i}. {step}")
    print()

    print("=== Тест 4: Дружба (транзитивность знакомства) ===")
    formulas4 = "Друг(Алиса, Боб), Друг(Боб, Чарли), ¬Друг(x,y) ∨ ¬Друг(y,z) ∨ Знаком(x,z), ¬Знаком(Алиса, Чарли)"
    proven4, steps4 = engine.prove(formulas4)
    print(f"Формулы: {formulas4}")
    print(f"Доказано: {proven4}")
    print("Шаги доказательства:")
    for i, step in enumerate(steps4, 1):
        print(f"{i}. {step}")
    print()

    print("=== Тест 5: Кто убил Агату? (классика логики) ===")
    formulas5 = ("Убийца(МиссСкарлетт) ∨ Убийца(ПолковникМастард) ∨ Убийца(МиссисПикок), "
                 "¬Убийца(МиссСкарлетт), "
                 "¬Убийца(ПолковникМастард), "
                 "¬Убийца(МиссисПикок)")
    proven5, steps5 = engine.prove(formulas5)
    print(f"Формулы: {formulas5}")
    print(f"Доказано: {proven5}")
    print("Шаги доказательства:")
    for i, step in enumerate(steps5, 1):
        print(f"{i}. {step}")
    print()

    print("=== Тест 6: Все рыцари лгут или все шпионы рыцари (парадокс) ===")
    formulas6 = ("¬Рыцарь(x) ∨ Лжёт(x), "           # Все рыцари лгут
                 "¬Шпион(x) ∨ Рыцарь(x), "           # Все шпионы — рыцари
                 "Существо(Ланселот), "
                 "Шпион(Ланселот), "
                 "¬Лжёт(Ланселот)")                  # Ланселот не лжёт → противоречие!
    proven6, steps6 = engine.prove(formulas6)
    print(f"Формулы: {formulas6}")
    print(f"Доказано: {proven6}")
    print("Шаги доказательства:")
    for i, step in enumerate(steps6, 1):
        print(f"{i}. {step}")
    print()
