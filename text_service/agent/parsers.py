import re
from typing import List, Dict


def parse_recipe_from_text(dish: str, text: str) -> Dict:
    recipe = {"title": dish, "ingredients": [], "steps": []}

    ingredient_patterns = [
        r"📋\s*재료[:\s]*([\s\S]*?)(?=👨‍🍳|조리법|👨|🍳|$)",
        r"재료[:\s]*([\s\S]*?)(?=조리법|👨‍🍳|👨|🍳|$)",
        r"•\s*([^•\n]*?)(?=\n|$)",
        r"재료[:\s]*([\s\S]*?)(?=\n\n|\n\d+\.|$)",
    ]

    for pattern in ingredient_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for match in matches:
                brace_matches = re.findall(r"\{'name':\s*'([^']+)',\s*'amount':\s*'([^']+)'\}", match)
                if brace_matches:
                    for name, amount in brace_matches:
                        recipe["ingredients"].append(f"{name} {amount}")
                else:
                    lines = match.strip().split("\n")
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith("•") and "재료" not in line:
                            recipe["ingredients"].append(line)
            break

    step_patterns = [
        r"👨‍🍳\s*조리법[:\s]*([\s\S]*?)(?=\n\n|$)",
        r"조리법[:\s]*([\s\S]*?)(?=\n\n|$)",
        r"(\d+\.\s*[^\n]+(?:\n\d+\.\s*[^\n]+)*)",
    ]

    for pattern in step_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for match in matches:
                lines = match.strip().split("\n")
                for line in lines:
                    line = line.strip()
                    if line and re.match(r"\d+\.", line):
                        step = re.sub(r"^\d+\.\s*\d+\.\s*", "", line)
                        step = re.sub(r"^\d+\.\s*", "", step)
                        if step:
                            recipe["steps"].append(step)
            break

    if len(recipe["steps"]) > 15:
        recipe["steps"] = recipe["steps"][:15]

    if not recipe["ingredients"]:
        recipe["ingredients"] = ["재료 정보를 찾을 수 없습니다"]
    if not recipe["steps"]:
        recipe["steps"] = ["조리법 정보를 찾을 수 없습니다"]

    return recipe


def parse_ingredients_from_text(dish: str, text: str) -> List[str]:
    ingredients: List[str] = []

    brace_matches = re.findall(r"\{'name':\s*'([^']+)',\s*'amount':\s*'([^']+)'(?:,\s*'unit':\s*'([^']+)')?\}", text)
    if brace_matches:
        for match in brace_matches:
            name = match[0]
            amount = match[1]
            unit = match[2] if len(match) > 2 and match[2] else ""
            if unit and unit != "''":
                ingredients.append(f"{name} {amount}{unit}")
            else:
                ingredients.append(f"{name} {amount}")
        return ingredients

    brace_matches_old = re.findall(r"\{'name':\s*'([^']+)',\s*'amount':\s*'([^']+)'\}", text)
    if brace_matches_old:
        for name, amount in brace_matches_old:
            ingredients.append(f"{name} {amount}")
        return ingredients

    ingredient_patterns = [
        r"📋\s*재료[:\s]*([\s\S]*?)(?=👨‍🍳|조리법|👨|🍳|$)",
        r"재료[:\s]*([\s\S]*?)(?=조리법|👨‍🍳|👨|🍳|$)",
        r"•\s*([^•\n]*?)(?=\n|$)",
        r"재료[:\s]*([\s\S]*?)(?=\n\n|\n\d+\.|$)",
    ]

    for pattern in ingredient_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for match in matches:
                lines = match.strip().split("\n")
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith("•") and "재료" not in line and "📋" not in line:
                        ingredients.append(line)
            break

    return ingredients if ingredients else ["재료 정보를 찾을 수 없습니다"]


def parse_tips_from_text(dish: str, text: str) -> List[str]:
    tips: List[str] = []

    tip_patterns = [
        r"💡\s*조리\s*팁[:\s]*([\s\S]*?)(?=\n\n|$)",
        r"조리\s*팁[:\s]*([\s\S]*?)(?=\n\n|$)",
        r"팁[:\s]*([\s\S]*?)(?=\n\n|$)",
        r"(\d+\.\s*[^\n]+(?:\n\d+\.\s*[^\n]+)*)",
    ]

    for pattern in tip_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for match in matches:
                lines = match.strip().split("\n")
                for line in lines:
                    line = line.strip()
                    if line and re.match(r"\d+\.", line):
                        tip = re.sub(r"^\d+\.\s*", "", line)
                        if tip and "팁" not in tip.lower():
                            tips.append(tip)
            break

    return tips if tips else ["조리 팁을 찾을 수 없습니다"]


