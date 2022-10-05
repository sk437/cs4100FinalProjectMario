import retro
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath("./supermarioretro"))


def main():
        retro.data.Integrations.add_custom_path(
                os.path.join(SCRIPT_DIR, "custom_integrations")
        )
        print("supermarioretro" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
        env = retro.make("supermarioretro", inttype=retro.data.Integrations.ALL)
        print(env)


if __name__ == "__main__":
        main()