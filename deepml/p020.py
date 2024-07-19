import numpy as np


def learn_decision_tree(
    examples: list[dict],
    attributes: list[str],
    target_attr: str,
) -> dict:
    # if we have exhausted our attributes or found an attribute
    # that uniquely identifies the target class, emit it
    if not attributes or len({e[target_attr] for e in examples}) == 1:
        return examples[0][target_attr]

    # compute total entropy of the current dataset
    h = entropy(examples, target_attr)

    # find the attribute that maximizes information gain,
    # which will be used to split the tree
    a = max(
        attributes,
        key=lambda a: sum(
            h - entropy([e for e in examples if e[a] == v], target_attr)
            for v in {e[a] for e in examples}
        ),
    )

    # split the tree and recur
    return {
        a: {
            v: learn_decision_tree(
                [e for e in examples if e[a] == v],
                [a2 for a2 in attributes if a2 != a],
                target_attr,
            )
            for v in {e[a] for e in examples}
        }
    }


def entropy(examples: list[dict], target_attr: str) -> float:
    return -sum(
        (pi := np.mean([e[target_attr] == c for e in examples]))
        * np.log2(pi.clip(1e-5))
        for c in {e[target_attr] for e in examples}
    )


def test_learn_decision_tree() -> None:
    examples = [
        {"Outlook": "Sunny", "Wind": "Weak", "PlayTennis": "No"},
        {"Outlook": "Overcast", "Wind": "Strong", "PlayTennis": "Yes"},
        {"Outlook": "Rain", "Wind": "Weak", "PlayTennis": "Yes"},
        {"Outlook": "Sunny", "Wind": "Strong", "PlayTennis": "No"},
        {"Outlook": "Sunny", "Wind": "Weak", "PlayTennis": "Yes"},
        {"Outlook": "Overcast", "Wind": "Weak", "PlayTennis": "Yes"},
        {"Outlook": "Rain", "Wind": "Strong", "PlayTennis": "No"},
        {"Outlook": "Rain", "Wind": "Weak", "PlayTennis": "Yes"},
    ]
    attributes = ["Outlook", "Wind"]
    target_attr = "PlayTennis"
    output = learn_decision_tree(examples, attributes, target_attr)
    expect = {
        "Outlook": {
            "Rain": {"Wind": {"Strong": "No", "Weak": "Yes"}},
            "Sunny": {"Wind": {"Strong": "No", "Weak": "No"}},
            "Overcast": "Yes",
        }
    }
    assert output == expect
