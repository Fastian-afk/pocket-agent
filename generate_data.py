"""
Phase 1: Synthetic Data Generation for Pocket-Agent
Hybrid approach: programmatic templates + Gemini 1.5 Flash enhancement.
Generates 2200+ ChatML-formatted training examples.
"""

import json
import hashlib
import os
import time
import random
import re
from pathlib import Path
from typing import List, Dict, Optional

# ── Configuration ──────────────────────────────────────────────────────────
OUTPUT_FILE = "train_data.jsonl"
PUBLIC_TEST_FILE = "public_test.jsonl"
GEMINI_MODEL = "gemini-1.5-flash"
TARGET_COUNTS = {
    "single_turn": 1800,
    "multi_turn": 350,
    "adversarial": 350,
    "refusal": 200,
}

SYSTEM_PROMPT = """You are Pocket-Agent, an on-device AI assistant with access to 5 tools.
When a user request matches a tool, respond ONLY with:
<tool_call>{"tool": "tool_name", "args": {...}}</tool_call>

Available tools:
1. weather — {"tool":"weather","args":{"location":"string","unit":"C|F"}}
2. calendar — {"tool":"calendar","args":{"action":"list|create","date":"YYYY-MM-DD","title":"string?"}}
3. convert — {"tool":"convert","args":{"value":number,"from_unit":"string","to_unit":"string"}}
4. currency — {"tool":"currency","args":{"amount":number,"from":"ISO3","to":"ISO3"}}
5. sql — {"tool":"sql","args":{"query":"string"}}

If no tool fits, respond with plain text only. Never invent tools."""


# ── Template Data ──────────────────────────────────────────────────────────
LOCATIONS = [
    "London", "Paris", "Tokyo", "New York", "Sydney", "Berlin", "Cairo",
    "Mumbai", "Toronto", "São Paulo", "Lagos", "Seoul", "Dubai", "Rome",
    "Bangkok", "Moscow", "Istanbul", "Lahore", "Karachi", "Islamabad",
    "Beijing", "Mexico City", "Nairobi", "Singapore", "Cape Town",
    "Buenos Aires", "Lima", "Jakarta", "Hanoi", "Athens", "Vienna",
    "Lisbon", "Oslo", "Helsinki", "Zurich", "Prague", "Warsaw",
    "Dublin", "Edinburgh", "Barcelona", "Riyadh", "Doha", "Kuala Lumpur",
    "Manila", "Bogotá", "Santiago", "Accra", "Casablanca", "Addis Ababa",
    "Timbuktu", "Reykjavik", "Anchorage", "Honolulu", "Kathmandu",
]

WEATHER_TEMPLATES = [
    "What's the weather in {loc}?",
    "Weather forecast for {loc} please",
    "How's the weather in {loc} right now?",
    "Tell me the temperature in {loc}",
    "Is it hot or cold in {loc}?",
    "What's the current temperature in {loc} in {unit_word}?",
    "I need the weather report for {loc}",
    "Can you check the weather in {loc}?",
    "Give me {loc} weather in {unit_word}",
    "I'm traveling to {loc} tomorrow, what's the weather like?",
    "What should I wear in {loc}? Check the weather",
    "How warm is it in {loc}?",
    "Check {loc} forecast",
    "Temperature in {loc}?",
    "Will it rain in {loc}?",
    "Weather update for {loc} in {unit_word} please",
    "{loc} weather",
    "What's it like outside in {loc}?",
    "Do I need a jacket in {loc}?",
    "Climate check for {loc}",
]

DATES = [
    "2025-01-15", "2025-02-14", "2025-03-01", "2025-03-20",
    "2025-04-10", "2025-05-01", "2025-06-15", "2025-07-04",
    "2025-08-20", "2025-09-10", "2025-10-31", "2025-11-25",
    "2025-12-25", "2025-12-31", "2025-01-01", "2025-06-01",
    "2025-04-23", "2025-09-30", "2025-07-20", "2025-11-11",
]

EVENT_TITLES = [
    "Team meeting", "Doctor appointment", "Dentist checkup",
    "Lunch with Sarah", "Project deadline", "Birthday party",
    "Gym session", "Flight to London", "Client presentation",
    "Date night", "Grocery shopping", "Car service",
    "Yoga class", "Book club", "Parent-teacher meeting",
    "Job interview", "Wedding anniversary", "Hackathon",
    "Conference call", "Movie night", "Study group",
    "Piano lesson", "Photography workshop", "Dinner reservation",
]

CALENDAR_LIST_TEMPLATES = [
    "What's on my calendar for {date}?",
    "Show me my events on {date}",
    "Any meetings on {date}?",
    "List my schedule for {date}",
    "What do I have planned for {date}?",
    "Check my calendar on {date}",
    "Am I free on {date}?",
    "What's happening on {date}?",
    "Do I have anything on {date}?",
    "Pull up my {date} schedule",
]

CALENDAR_CREATE_TEMPLATES = [
    "Add '{title}' to my calendar on {date}",
    "Schedule '{title}' for {date}",
    "Create an event '{title}' on {date}",
    "Put '{title}' on {date}",
    "Book '{title}' for {date}",
    "Set up '{title}' on {date}",
    "Remind me about '{title}' on {date}",
    "I need to add '{title}' on {date}",
    "New event: '{title}' on {date}",
    "Can you schedule '{title}' on {date} for me?",
]

CONVERSION_PAIRS = [
    (5, "miles", "kilometers"), (10, "kilometers", "miles"),
    (100, "pounds", "kilograms"), (50, "kilograms", "pounds"),
    (72, "Fahrenheit", "Celsius"), (25, "Celsius", "Fahrenheit"),
    (6, "feet", "meters"), (180, "centimeters", "inches"),
    (1, "gallon", "liters"), (5, "liters", "gallons"),
    (200, "grams", "ounces"), (16, "ounces", "grams"),
    (1, "mile", "feet"), (100, "meters", "yards"),
    (2.5, "inches", "centimeters"), (1000, "milliliters", "liters"),
    (3, "tablespoons", "teaspoons"), (1, "cup", "milliliters"),
    (60, "minutes", "hours"), (7, "days", "hours"),
    (500, "meters", "miles"), (12, "inches", "feet"),
    (30, "Celsius", "Kelvin"), (350, "Fahrenheit", "Celsius"),
    (1, "ton", "kilograms"), (2.2, "pounds", "kilograms"),
    (100, "yards", "meters"), (1, "nautical mile", "kilometers"),
    (8, "cups", "liters"), (32, "fluid ounces", "milliliters"),
]

CONVERT_TEMPLATES = [
    "Convert {val} {from_u} to {to_u}",
    "How many {to_u} is {val} {from_u}?",
    "What's {val} {from_u} in {to_u}?",
    "{val} {from_u} to {to_u}",
    "How much is {val} {from_u} in {to_u}?",
    "Change {val} {from_u} to {to_u}",
    "Please convert {val} {from_u} into {to_u}",
    "I need {val} {from_u} converted to {to_u}",
    "What does {val} {from_u} equal in {to_u}?",
    "Tell me {val} {from_u} in {to_u}",
]

CURRENCIES = [
    "USD", "EUR", "GBP", "JPY", "PKR", "INR", "CAD", "AUD",
    "CHF", "CNY", "KRW", "BRL", "MXN", "SGD", "HKD", "NOK",
    "SEK", "DKK", "NZD", "ZAR", "TRY", "SAR", "AED", "THB",
]

CURRENCY_TEMPLATES = [
    "Convert {amt} {from_c} to {to_c}",
    "How much is {amt} {from_c} in {to_c}?",
    "What's {amt} {from_c} worth in {to_c}?",
    "{amt} {from_c} to {to_c}",
    "Exchange {amt} {from_c} to {to_c}",
    "I have {amt} {from_c}, how much in {to_c}?",
    "Give me {amt} {from_c} in {to_c}",
    "How many {to_c} for {amt} {from_c}?",
    "Change {amt} {from_c} into {to_c}",
    "What's the value of {amt} {from_c} in {to_c}?",
]

SQL_QUERIES = [
    ("Show me all users", "SELECT * FROM users"),
    ("Get all orders from last month", "SELECT * FROM orders WHERE date >= DATE_SUB(NOW(), INTERVAL 1 MONTH)"),
    ("How many customers do we have?", "SELECT COUNT(*) FROM customers"),
    ("List products under $50", "SELECT * FROM products WHERE price < 50"),
    ("Find the top 5 sellers", "SELECT seller_id, SUM(amount) as total FROM sales GROUP BY seller_id ORDER BY total DESC LIMIT 5"),
    ("Show employees in the engineering department", "SELECT * FROM employees WHERE department = 'engineering'"),
    ("Average order value", "SELECT AVG(total) FROM orders"),
    ("Get all active subscriptions", "SELECT * FROM subscriptions WHERE status = 'active'"),
    ("Total revenue this year", "SELECT SUM(revenue) FROM transactions WHERE YEAR(date) = 2025"),
    ("Users who signed up today", "SELECT * FROM users WHERE DATE(created_at) = CURDATE()"),
    ("List all tables", "SHOW TABLES"),
    ("Count orders by status", "SELECT status, COUNT(*) FROM orders GROUP BY status"),
    ("Find duplicate emails", "SELECT email, COUNT(*) as cnt FROM users GROUP BY email HAVING cnt > 1"),
    ("Get user with id 42", "SELECT * FROM users WHERE id = 42"),
    ("Delete inactive users", "DELETE FROM users WHERE last_login < DATE_SUB(NOW(), INTERVAL 1 YEAR)"),
    ("Update product price", "UPDATE products SET price = 29.99 WHERE id = 101"),
    ("Join orders with customers", "SELECT o.*, c.name FROM orders o JOIN customers c ON o.customer_id = c.id"),
    ("Find the cheapest product", "SELECT * FROM products ORDER BY price ASC LIMIT 1"),
    ("Monthly sales report", "SELECT MONTH(date) as month, SUM(amount) FROM sales GROUP BY MONTH(date)"),
    ("Get all pending invoices", "SELECT * FROM invoices WHERE status = 'pending'"),
    ("Count users by country", "SELECT country, COUNT(*) FROM users GROUP BY country ORDER BY COUNT(*) DESC"),
    ("Search products by name", "SELECT * FROM products WHERE name LIKE '%phone%'"),
    ("Latest 10 transactions", "SELECT * FROM transactions ORDER BY created_at DESC LIMIT 10"),
    ("Total inventory value", "SELECT SUM(price * quantity) FROM inventory"),
    ("Users with no orders", "SELECT u.* FROM users u LEFT JOIN orders o ON u.id = o.user_id WHERE o.id IS NULL"),
]

SQL_TEMPLATES = [
    "{query_nl}",
    "Run this query: {query_nl}",
    "I need a SQL query to {query_nl_lower}",
    "Can you write SQL for: {query_nl_lower}",
    "Execute: {query_nl_lower}",
    "Query the database: {query_nl_lower}",
    "SQL for {query_nl_lower}",
    "Pull data: {query_nl_lower}",
]

REFUSAL_QUERIES = [
    "Tell me a joke", "What's the meaning of life?", "How are you today?",
    "Who won the World Cup in 2022?", "Write me a poem about the ocean",
    "What's your name?", "Can you sing a song?", "Tell me a bedtime story",
    "What's 2 + 2?", "Explain quantum physics", "Who is the president of USA?",
    "Book me a flight to Dubai", "Send an email to John",
    "Order me a pizza", "Play some music", "Set an alarm for 7 AM",
    "Take a screenshot", "Open my camera", "Call my mom",
    "What's the capital of France?", "Translate hello to French",
    "How do I cook pasta?", "What's the best phone to buy?",
    "Recommend a good movie", "What time is it?",
    "How old are you?", "Are you smarter than Siri?",
    "Can you hack into NASA?", "Give me lottery numbers",
    "What's the stock price of Apple?", "Read my messages",
    "Turn off the lights", "Navigate to the nearest gas station",
    "Find me a restaurant nearby", "What's trending on Twitter?",
    "Summarize this article for me", "Debug this Python code",
    "Make me a website", "What's my horoscope?",
    "How do I lose weight?", "What's the weather going to be like next month everywhere?",
    "Can you browse the internet for me?", "Download this file",
    "Scan this QR code", "What's the WiFi password?",
    "Remind me to drink water every hour", "Track my package",
    "Find cheap flights", "Show me memes",
    "Hi there!", "Good morning!", "Thanks a lot!", "Bye!",
    "You're awesome", "I'm bored", "Do you dream?",
    "What are you?", "Can you learn?", "Help me study",
    "What's your favorite color?", "I feel sad today",
    "Motivate me", "Tell me something interesting",
    "What day is it?", "How many planets are there?",
    "Who invented the telephone?", "What's DNA?",
    "Can you drive a car?", "Do you eat food?",
    "Plan me a vacation to Mars", "Beam me up Scotty",
    "Use the translator tool", "Activate my smart home",
    "Search Google for Python tutorials",
    "Use the calculator tool to add 5 and 3",
    "Open the maps tool and find directions",
    "Use the email tool to send a message",
    "Access the file manager tool",
    "Launch the music player tool",
]

REFUSAL_RESPONSES = [
    "I don't have a tool for that, but {followup}",
    "That's outside my tool capabilities. {followup}",
    "I can't do that with my available tools, but {followup}",
    "I'm not able to help with that specific request. {followup}",
    "That's not something my tools can handle. {followup}",
]

REFUSAL_FOLLOWUPS = [
    "I can help with weather, calendar, unit conversion, currency exchange, or SQL queries!",
    "Feel free to ask me about weather, calendars, conversions, currencies, or database queries.",
    "I'm great at weather lookups, calendar management, conversions, currency exchange, and SQL!",
    "Try asking me to check the weather, manage your calendar, convert units, exchange currency, or run SQL.",
    "My tools include weather, calendar, convert, currency, and sql — want to try one?",
    "I can help you with weather checks, scheduling, unit conversions, currency exchanges, or SQL queries.",
    "Let me know if you need help with any of my available tools: weather, calendar, convert, currency, or sql.",
]

# ── Adversarial Templates ─────────────────────────────────────────────────
TYPO_MAP = {
    "weather": ["weathr", "wether", "wheather", "waether", "wetaher"],
    "convert": ["conver", "covnert", "convetr", "cnvert"],
    "calendar": ["calender", "calandar", "calander", "calnedar"],
    "currency": ["currnecy", "curency", "currenxy", "currancy"],
}

TYPO_LOCATIONS = {
    "London": ["Lndon", "Londno", "Londn"],
    "Paris": ["Pars", "Prais", "Parsi"],
    "Tokyo": ["Tokoy", "Tokyp", "Tokoyo"],
    "New York": ["New Yrok", "Nwe York", "New Yor"],
    "Lahore": ["Lahor", "Lahoer", "Lahroe"],
    "Karachi": ["Karahci", "Karchi", "Karacih"],
}

CODE_SWITCH_TEMPLATES = [
    # Urdu-English
    ("Mujhe {loc} ka weather batao", "weather", {"location": "{loc}", "unit": "C"}),
    ("{loc} mein temperature kya hai?", "weather", {"location": "{loc}", "unit": "C"}),
    ("Aaj {loc} ka mausam kaisa hai?", "weather", {"location": "{loc}", "unit": "C"}),
    ("{loc} mein garmi hai ya sardi?", "weather", {"location": "{loc}", "unit": "C"}),
    ("Kal ke liye {date} pe meeting schedule karo", "calendar", {"action": "create", "date": "{date}", "title": "meeting"}),
    ("{val} dollars ko rupees mein convert karo", "currency", {"amount": "{val}", "from": "USD", "to": "PKR"}),
    ("{val} {from_u} ko {to_u} mein badlo", "convert", {"value": "{val}", "from_unit": "{from_u}", "to_unit": "{to_u}"}),
    # Spanish-English
    ("Dime el weather en {loc}", "weather", {"location": "{loc}", "unit": "C"}),
    ("¿Cuál es el clima en {loc}?", "weather", {"location": "{loc}", "unit": "C"}),
    ("Convierte {val} {from_u} a {to_u}", "convert", {"value": "{val}", "from_unit": "{from_u}", "to_unit": "{to_u}"}),
    ("¿Cuánto es {val} {from_c} en {to_c}?", "currency", {"amount": "{val}", "from": "{from_c}", "to": "{to_c}"}),
    # Arabic-English
    ("أعطني weather في {loc}", "weather", {"location": "{loc}", "unit": "C"}),
    ("كم يساوي {val} {from_c} بال {to_c}؟", "currency", {"amount": "{val}", "from": "{from_c}", "to": "{to_c}"}),
    # Hindi-English
    ("{loc} ka weather kya hai?", "weather", {"location": "{loc}", "unit": "C"}),
    ("{val} {from_u} kitne {to_u} hote hain?", "convert", {"value": "{val}", "from_unit": "{from_u}", "to_unit": "{to_u}"}),
]

# ── Helpers ────────────────────────────────────────────────────────────────

def make_tool_call(tool: str, args: dict) -> str:
    """Format a tool call response."""
    return f'<tool_call>{json.dumps({"tool": tool, "args": args}, ensure_ascii=False)}</tool_call>'


def make_example(user_msg: str, assistant_msg: str, system: str = SYSTEM_PROMPT) -> dict:
    """Create a single-turn training example."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def make_multi_turn(turns: List[Dict], system: str = SYSTEM_PROMPT) -> dict:
    """Create a multi-turn training example. turns = [{'user':..,'assistant':..}, ...]"""
    messages = [{"role": "system", "content": system}]
    for t in turns:
        messages.append({"role": "user", "content": t["user"]})
        messages.append({"role": "assistant", "content": t["assistant"]})
    return {"messages": messages}


def sha256(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def load_public_hashes() -> set:
    """Load SHA-256 hashes of public test prompts for dedup."""
    hashes = set()
    if Path(PUBLIC_TEST_FILE).exists():
        with open(PUBLIC_TEST_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    prompt = obj.get("prompt", obj.get("query", ""))
                    if prompt:
                        hashes.add(sha256(prompt))
                except json.JSONDecodeError:
                    continue
    print(f"Loaded {len(hashes)} public test hashes for dedup")
    return hashes


# ── Generators ─────────────────────────────────────────────────────────────

def generate_weather_examples(count: int) -> List[dict]:
    examples = []
    for _ in range(count):
        loc = random.choice(LOCATIONS)
        unit = random.choice(["C", "F"])
        unit_word = "Celsius" if unit == "C" else "Fahrenheit"
        tmpl = random.choice(WEATHER_TEMPLATES)
        query = tmpl.format(loc=loc, unit_word=unit_word)
        resp = make_tool_call("weather", {"location": loc, "unit": unit})
        examples.append(make_example(query, resp))
    return examples


def generate_calendar_examples(count: int) -> List[dict]:
    examples = []
    half = count // 2
    # List events
    for _ in range(half):
        date = random.choice(DATES)
        tmpl = random.choice(CALENDAR_LIST_TEMPLATES)
        query = tmpl.format(date=date)
        resp = make_tool_call("calendar", {"action": "list", "date": date})
        examples.append(make_example(query, resp))
    # Create events
    for _ in range(count - half):
        date = random.choice(DATES)
        title = random.choice(EVENT_TITLES)
        tmpl = random.choice(CALENDAR_CREATE_TEMPLATES)
        query = tmpl.format(date=date, title=title)
        resp = make_tool_call("calendar", {"action": "create", "date": date, "title": title})
        examples.append(make_example(query, resp))
    return examples


def generate_convert_examples(count: int) -> List[dict]:
    examples = []
    for _ in range(count):
        val, from_u, to_u = random.choice(CONVERSION_PAIRS)
        val = round(val * random.uniform(0.5, 3.0), 1)
        tmpl = random.choice(CONVERT_TEMPLATES)
        query = tmpl.format(val=val, from_u=from_u, to_u=to_u)
        resp = make_tool_call("convert", {"value": val, "from_unit": from_u, "to_unit": to_u})
        examples.append(make_example(query, resp))
    return examples


def generate_currency_examples(count: int) -> List[dict]:
    examples = []
    for _ in range(count):
        from_c, to_c = random.sample(CURRENCIES, 2)
        amt = round(random.choice([1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000]) * random.uniform(0.8, 1.5), 2)
        tmpl = random.choice(CURRENCY_TEMPLATES)
        query = tmpl.format(amt=amt, from_c=from_c, to_c=to_c)
        resp = make_tool_call("currency", {"amount": amt, "from": from_c, "to": to_c})
        examples.append(make_example(query, resp))
    return examples


def generate_sql_examples(count: int) -> List[dict]:
    examples = []
    for _ in range(count):
        query_nl, sql = random.choice(SQL_QUERIES)
        tmpl = random.choice(SQL_TEMPLATES)
        query = tmpl.format(query_nl=query_nl, query_nl_lower=query_nl.lower())
        resp = make_tool_call("sql", {"query": sql})
        examples.append(make_example(query, resp))
    return examples


def generate_multi_turn_examples(count: int) -> List[dict]:
    """Generate multi-turn conversations with context references."""
    examples = []
    patterns = [
        # Weather → Convert temperature
        lambda: {
            "turns": [
                {"user": f"What's the weather in {(loc := random.choice(LOCATIONS))}?",
                 "assistant": make_tool_call("weather", {"location": loc, "unit": "C"})},
                {"user": "Convert that temperature to Fahrenheit",
                 "assistant": make_tool_call("convert", {"value": 25, "from_unit": "Celsius", "to_unit": "Fahrenheit"})},
            ]
        },
        # Currency → Currency again
        lambda: {
            "turns": [
                {"user": f"How much is {(amt := random.randint(10, 500))} {(fc := random.choice(CURRENCIES[:8]))} in {(tc := random.choice(CURRENCIES[8:16]))}?",
                 "assistant": make_tool_call("currency", {"amount": amt, "from": fc, "to": tc})},
                {"user": f"Now convert that to {(tc2 := random.choice([c for c in CURRENCIES if c not in [fc, tc]]))}",
                 "assistant": make_tool_call("currency", {"amount": amt, "from": fc, "to": tc2})},
            ]
        },
        # Convert → Convert different unit
        lambda: {
            "turns": [
                {"user": f"Convert {(v := random.randint(1, 100))} miles to kilometers",
                 "assistant": make_tool_call("convert", {"value": v, "from_unit": "miles", "to_unit": "kilometers"})},
                {"user": "Now convert that to meters",
                 "assistant": make_tool_call("convert", {"value": v, "from_unit": "miles", "to_unit": "meters"})},
            ]
        },
        # Calendar list → Calendar create
        lambda: {
            "turns": [
                {"user": f"What's on my calendar for {(d := random.choice(DATES))}?",
                 "assistant": make_tool_call("calendar", {"action": "list", "date": d})},
                {"user": f"Add '{(t := random.choice(EVENT_TITLES))}' to that day",
                 "assistant": make_tool_call("calendar", {"action": "create", "date": d, "title": t})},
            ]
        },
        # Weather → Weather different location
        lambda: {
            "turns": [
                {"user": f"Weather in {random.choice(LOCATIONS[:20])}?",
                 "assistant": make_tool_call("weather", {"location": random.choice(LOCATIONS[:20]), "unit": "C"})},
                {"user": f"What about {(loc2 := random.choice(LOCATIONS[20:]))}?",
                 "assistant": make_tool_call("weather", {"location": loc2, "unit": "C"})},
            ]
        },
        # Convert → Currency
        lambda: {
            "turns": [
                {"user": f"How many kilograms is {(v := random.randint(50, 200))} pounds?",
                 "assistant": make_tool_call("convert", {"value": v, "from_unit": "pounds", "to_unit": "kilograms"})},
                {"user": f"Also convert {random.randint(10, 100)} USD to EUR",
                 "assistant": make_tool_call("currency", {"amount": random.randint(10, 100), "from": "USD", "to": "EUR"})},
            ]
        },
    ]

    for i in range(count):
        pattern = patterns[i % len(patterns)]
        data = pattern()
        examples.append(make_multi_turn(data["turns"]))
    return examples


def generate_adversarial_examples(count: int) -> List[dict]:
    """Generate adversarial examples: typos, code-switching, hallucination baits."""
    examples = []
    third = count // 3

    # Typo examples
    for _ in range(third):
        loc = random.choice(list(TYPO_LOCATIONS.keys()))
        typo_loc = random.choice(TYPO_LOCATIONS[loc])
        unit = random.choice(["C", "F"])
        queries = [
            f"What's the weathr in {typo_loc}?",
            f"Chekc the weather in {typo_loc}",
            f"Wheather forcast for {typo_loc}",
            f"Temprature in {typo_loc}?",
            f"Waht is the weather lke in {typo_loc}?",
            f"Wether in {typo_loc} plz",
            f"Hows the wether in {typo_loc}",
            f"Tell me abt weather in {typo_loc}",
        ]
        query = random.choice(queries)
        resp = make_tool_call("weather", {"location": loc, "unit": unit})
        examples.append(make_example(query, resp))

    # Code-switching examples
    for _ in range(third):
        tmpl_data = random.choice(CODE_SWITCH_TEMPLATES)
        query_tmpl, tool, args_tmpl = tmpl_data
        loc = random.choice(LOCATIONS)
        val = random.randint(1, 1000)
        from_u, to_u = "miles", "kilometers"
        if random.random() > 0.5:
            from_u, to_u = "pounds", "kilograms"
        from_c, to_c = random.sample(CURRENCIES[:8], 2)
        date = random.choice(DATES)

        query = query_tmpl.format(
            loc=loc, val=val, from_u=from_u, to_u=to_u,
            from_c=from_c, to_c=to_c, date=date
        )
        args = {}
        for k, v in args_tmpl.items():
            v_str = str(v)
            v_str = v_str.replace("{loc}", loc).replace("{val}", str(val))
            v_str = v_str.replace("{from_u}", from_u).replace("{to_u}", to_u)
            v_str = v_str.replace("{from_c}", from_c).replace("{to_c}", to_c)
            v_str = v_str.replace("{date}", date)
            try:
                args[k] = json.loads(v_str)
            except (json.JSONDecodeError, ValueError):
                args[k] = v_str

        resp = make_tool_call(tool, args)
        examples.append(make_example(query, resp))

    # Hallucination bait examples (valid tool calls for fictional/unusual places)
    bait_locations = [
        "Atlantis", "Hogwarts", "Mordor", "Wakanda", "Narnia",
        "Gotham City", "Springfield", "Bikini Bottom", "Asgard",
        "The Shire", "Neverland", "El Dorado", "Shangri-La",
    ]
    for _ in range(count - 2 * third):
        loc = random.choice(bait_locations)
        unit = random.choice(["C", "F"])
        queries = [
            f"What's the weather in {loc}?",
            f"Check weather for {loc}",
            f"Temperature in {loc}?",
            f"Is it cold in {loc}?",
        ]
        query = random.choice(queries)
        resp = make_tool_call("weather", {"location": loc, "unit": unit})
        examples.append(make_example(query, resp))

    return examples


def generate_refusal_examples(count: int) -> List[dict]:
    """Generate refusal examples — plain text responses, NO tool calls."""
    examples = []
    # Use all unique queries first
    for i, query in enumerate(REFUSAL_QUERIES):
        followup = random.choice(REFUSAL_FOLLOWUPS)
        if "joke" in query.lower():
            resp = "I appreciate the request, but I don't have a jokes tool! " + followup
        elif "flight" in query.lower() or "email" in query.lower() or "pizza" in query.lower():
            resp = f"I don't have a tool for that. {followup}"
        elif "hi" in query.lower() or "hello" in query.lower() or "morning" in query.lower() or "bye" in query.lower() or "thanks" in query.lower():
            greetings = ["Hello! How can I help you today?", "Hi there! What can I do for you?", "Hey! Need help with anything?", "You're welcome! Let me know if you need anything else.", "Goodbye! Have a great day!"]
            resp = random.choice(greetings) + " " + followup
        elif any(w in query.lower() for w in ["tool", "activate", "launch", "access"]):
            resp = f"I don't have that tool. {followup}"
        else:
            resp_tmpl = random.choice(REFUSAL_RESPONSES)
            resp = resp_tmpl.format(followup=followup)
        examples.append(make_example(query, resp))
    # Generate paraphrased variants for remaining count
    paraphrase_prefixes = [
        "Hey, ", "Could you ", "Please ", "I want you to ", "Can you please ",
        "I'd like you to ", "Would you ", "I need you to ", "Yo, ", "Umm, ",
    ]
    remaining = count - len(REFUSAL_QUERIES)
    for i in range(max(0, remaining)):
        base_query = REFUSAL_QUERIES[i % len(REFUSAL_QUERIES)]
        prefix = paraphrase_prefixes[i % len(paraphrase_prefixes)]
        query = prefix + base_query[0].lower() + base_query[1:]
        followup = random.choice(REFUSAL_FOLLOWUPS)
        resp_tmpl = random.choice(REFUSAL_RESPONSES)
        resp = resp_tmpl.format(followup=followup)
        examples.append(make_example(query, resp))
    return examples


# ── Gemini Enhancement (Optional) ─────────────────────────────────────────

def enhance_with_gemini(examples: List[dict], api_key: str) -> List[dict]:
    """Use Gemini to generate additional diverse examples."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
    except Exception as e:
        print(f"⚠ Gemini not available: {e}. Using template data only.")
        return examples

    extra = []
    prompts = [
        # Extra adversarial
        f"""Generate 50 JSON objects for tool-calling training. Each object:
{{"query": "user query with typos/slang/code-switching", "tool": "weather|calendar|convert|currency|sql", "args": {{...}}}}

Mix languages: English+Urdu, English+Spanish, English+Arabic, English+Hindi.
Include intentional typos. The model should still identify the correct tool.
Tool schemas: weather(location,unit:C|F), calendar(action:list|create,date:YYYY-MM-DD,title?), convert(value,from_unit,to_unit), currency(amount,from:ISO3,to:ISO3), sql(query).
Output ONLY a JSON array.""",

        # Extra refusals
        f"""Generate 50 JSON objects of queries that should NOT trigger any tool call.
{{"query": "user message", "response": "helpful plain text declining and suggesting available tools"}}

Available tools: weather, calendar, convert, currency, sql.
Include: chitchat, requests for nonexistent tools (flight booking, email, translation, image generation, web search, alarm), general knowledge, emotional support.
Output ONLY a JSON array.""",

        # Extra multi-turn
        f"""Generate 30 multi-turn conversations for a tool-calling assistant.
Each conversation: {{"turns": [{{"user": "...", "tool": "...", "args": {{...}}}}, {{"user": "follow-up referencing previous turn", "tool": "...", "args": {{...}}}}]}}

The second turn MUST reference the first ("convert that", "now in euros", "what about tomorrow", "same but in kilometers").
Tools: weather(location,unit:C|F), calendar(action:list|create,date,title?), convert(value,from_unit,to_unit), currency(amount,from:ISO3,to:ISO3), sql(query).
Output ONLY a JSON array.""",
    ]

    for i, prompt in enumerate(prompts):
        try:
            print(f"  Gemini batch {i+1}/{len(prompts)}...")
            response = model.generate_content(prompt)
            text = response.text.strip()
            # Extract JSON array
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                items = json.loads(match.group())
                for item in items:
                    if "query" in item and "tool" in item and "args" in item:
                        resp = make_tool_call(item["tool"], item["args"])
                        extra.append(make_example(item["query"], resp))
                    elif "query" in item and "response" in item:
                        extra.append(make_example(item["query"], item["response"]))
                    elif "turns" in item:
                        turns = []
                        for t in item["turns"]:
                            turns.append({
                                "user": t["user"],
                                "assistant": make_tool_call(t["tool"], t["args"]) if "tool" in t else t.get("assistant", "")
                            })
                        if turns:
                            extra.append(make_multi_turn(turns))
                print(f"    → Got {len(items)} examples")
            time.sleep(4)  # Rate limiting
        except Exception as e:
            print(f"    ⚠ Batch {i+1} failed: {e}")
            continue

    return examples + extra


# ── Main Pipeline ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Pocket-Agent — Synthetic Data Generation")
    print("=" * 60)

    random.seed(42)
    public_hashes = load_public_hashes()

    # Generate template-based data
    print("\n📊 Generating template-based examples...")
    all_examples = []

    print(f"  Weather examples ({TARGET_COUNTS['single_turn'] // 5})...")
    all_examples.extend(generate_weather_examples(TARGET_COUNTS["single_turn"] // 5))

    print(f"  Calendar examples ({TARGET_COUNTS['single_turn'] // 5})...")
    all_examples.extend(generate_calendar_examples(TARGET_COUNTS["single_turn"] // 5))

    print(f"  Convert examples ({TARGET_COUNTS['single_turn'] // 5})...")
    all_examples.extend(generate_convert_examples(TARGET_COUNTS["single_turn"] // 5))

    print(f"  Currency examples ({TARGET_COUNTS['single_turn'] // 5})...")
    all_examples.extend(generate_currency_examples(TARGET_COUNTS["single_turn"] // 5))

    print(f"  SQL examples ({TARGET_COUNTS['single_turn'] // 5})...")
    all_examples.extend(generate_sql_examples(TARGET_COUNTS["single_turn"] // 5))

    print(f"  Multi-turn examples ({TARGET_COUNTS['multi_turn']})...")
    all_examples.extend(generate_multi_turn_examples(TARGET_COUNTS["multi_turn"]))

    print(f"  Adversarial examples ({TARGET_COUNTS['adversarial']})...")
    all_examples.extend(generate_adversarial_examples(TARGET_COUNTS["adversarial"]))

    print(f"  Refusal examples ({TARGET_COUNTS['refusal']})...")
    all_examples.extend(generate_refusal_examples(TARGET_COUNTS["refusal"]))

    print(f"\n  Total template examples: {len(all_examples)}")

    # Optional Gemini enhancement
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        print("\n🤖 Enhancing with Gemini 1.5 Flash...")
        all_examples = enhance_with_gemini(all_examples, api_key)
        print(f"  Total after Gemini: {len(all_examples)}")
    else:
        print("\n⚠ No GEMINI_API_KEY set. Using template data only.")
        print("  Set it with: set GEMINI_API_KEY=your_key_here")

    # SHA-256 dedup against public test set
    print("\n🔒 Running SHA-256 dedup...")
    deduped = []
    seen_hashes = set()
    removed = 0
    for ex in all_examples:
        user_msgs = [m["content"] for m in ex["messages"] if m["role"] == "user"]
        prompt_hash = sha256(" ".join(user_msgs))
        if prompt_hash in public_hashes:
            removed += 1
            continue
        if prompt_hash in seen_hashes:
            removed += 1
            continue
        seen_hashes.add(prompt_hash)
        deduped.append(ex)

    print(f"  Removed {removed} duplicates")
    print(f"  Final dataset: {len(deduped)} examples")

    # Shuffle
    random.shuffle(deduped)

    # Write output
    print(f"\n💾 Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in deduped:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Stats
    tool_counts = {}
    refusals = 0
    multi_turn = 0
    for ex in deduped:
        msgs = ex["messages"]
        if len(msgs) > 3:
            multi_turn += 1
        for m in msgs:
            if m["role"] == "assistant":
                if "<tool_call>" in m["content"]:
                    try:
                        tc = json.loads(m["content"].split("<tool_call>")[1].split("</tool_call>")[0])
                        tool_counts[tc["tool"]] = tool_counts.get(tc["tool"], 0) + 1
                    except:
                        pass
                else:
                    refusals += 1

    print("\n📈 Dataset Statistics:")
    print(f"  Total examples: {len(deduped)}")
    print(f"  Multi-turn: {multi_turn}")
    print(f"  Refusals: {refusals}")
    print(f"  Tool calls by tool:")
    for tool, count in sorted(tool_counts.items()):
        print(f"    {tool}: {count}")

    print("\n✅ Data generation complete!")


if __name__ == "__main__":
    main()
