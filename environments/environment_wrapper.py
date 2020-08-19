import json

def get_env(name, config={}):
    with open("data/%s/config.json" % env_name) as f:
        config = json.load(f)