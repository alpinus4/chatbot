import json
from chat import Chat

import config as c

def main():
    passed = 0
    total = 0
    chat = Chat()
    chat.load_data()
    with open(c.TEST_DATA[0], 'r') as f:
            test_data = json.load(f)
    with open(c.INTENTS_DATA[0], 'r') as f:
            train_data = json.load(f)
    for i, test in enumerate(test_data["test"]):
        for pattern in test["patterns"]:
            response = chat.get_response(pattern)
            total += 1
            if response in train_data["intents"][i]["responses"]:
                print(f"{test['tag']} -- PASS")
                passed += 1
            else:
                print(f"{test['tag']} -- FAIL")
    print()
    print(f"Pass rate: {passed / total}")
    


if __name__ == "__main__":
    main()