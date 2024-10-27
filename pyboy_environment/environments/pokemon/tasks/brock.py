from functools import cached_property

import numpy as np
from pyboy.utils import WindowEvent

from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)
from pyboy_environment.environments.pokemon import pokemon_constants as pkc


class PokemonBrock(PokemonEnvironment):
    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 0,
        headless: bool = False,
    ) -> None:

        self.prev_gamestate = None
        self.gamestate = None
        self.current_location = None
        self.prev_location = None
        self.enemy_hp = None
        self.is_in_battle = False
        self.distance = 0
        self.prev_locations = {}
        self.prev_maps = {}

        # Define mappings for each category
        TRAINER_PAIRS = [[128, 139], [0, 11]]
        WALKABLE_PAIRS = [[257], [259], [278], [273], [291], [282, 284], [313], [334, 335], [341, 343], [300, 300]]
        GRASS_PAIRS = [[338, 338]]
        DOOR_MARKERS = [[260], [262], [267], [276]]
        self.game_area_mapping = {}
        self._add_to_mapping(TRAINER_PAIRS, 2)
        self._add_to_mapping(WALKABLE_PAIRS, 0)
        self._add_to_mapping(GRASS_PAIRS, 4)
        self._add_to_mapping(DOOR_MARKERS, 3)

        # Arrays for battle options to be avoided
        self.pokemon_select = np.array([
            [383, 311, 318, 325, 332, 339, 346, 353, 383, 367, 374, 374, 374, 374, 374, 374, 374, 374, 375, 383],
            [377, 378, 378, 378, 378, 378, 378, 378, 377, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 379],
            [380, 383, 383, 383, 383, 383, 383, 383, 380, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 380],
            [380, 383, 383, 383, 383, 383, 383, 383, 380, 383, 133, 136, 134, 135, 147, 237, 225, 226, 383, 380],
            [380, 383, 383, 383, 383, 383, 383, 383, 380, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 380],
            [380, 383, 383, 383, 383, 383, 383, 383, 380, 383, 136, 147, 132, 140, 383, 383, 145, 148, 141, 380],
            [381, 378, 378, 378, 378, 378, 378, 378, 381, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 382]
        ])
        self.battle_flee = np.array([
            [383, 311, 318, 325, 332, 339, 346, 353, 383, 367, 374, 374, 374, 374, 374, 374, 374, 374, 375, 383],
            [377, 378, 378, 378, 378, 378, 378, 378, 377, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 379],
            [380, 383, 383, 383, 383, 383, 383, 383, 380, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 380],
            [380, 383, 383, 383, 383, 383, 383, 383, 380, 383, 133, 136, 134, 135, 147, 383, 225, 226, 383, 380],
            [380, 383, 383, 383, 383, 383, 383, 383, 380, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 380],
            [380, 383, 383, 383, 383, 383, 383, 383, 380, 383, 136, 147, 132, 140, 383, 237, 145, 148, 141, 380],
            [381, 378, 378, 378, 378, 378, 378, 378, 381, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 382]
        ])

        valid_actions: list[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            #WindowEvent.PRESS_BUTTON_START,
        ]

        release_button: list[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            #WindowEvent.RELEASE_BUTTON_START,
        ]

        super().__init__(
            act_freq=act_freq,
            task="brock",
            init_name="has_pokedex.state",
            emulation_speed=emulation_speed,
            valid_actions=valid_actions,
            release_button=release_button,
            headless=headless,
        )

    ### STATE RELATED FUNCTIONS

    def _add_to_mapping(self, pairs, category):
        for pair in pairs:
            if len(pair) == 2:
                for value in range(pair[0], pair[1] + 1):
                    self.game_area_mapping[value] = category
            else:
                self.game_area_mapping[pair[0]] = category

    def _simplify_tile(self, value):
        return self.game_area_mapping.get(value, 1)

    def _simplify_game_area(self) -> np.ndarray:
        game_area = PokemonEnvironment.game_area(self)
        simplied_area = np.zeros(game_area.size, dtype = int)
        for i in range(game_area.size):
            simplied_area[i] = self._simplify_tile(game_area.flat[i])
        return simplied_area

    def _get_state(self) -> np.ndarray:
        game_stats = self._generate_game_stats()

        self.gamestate = game_stats
        self.current_location = (game_stats["location"]["x"], game_stats["location"]["y"], game_stats["location"]["map_id"])
        if (self.current_location[2] in self.prev_maps):
            self.distance = np.sqrt((self.current_location[0] - self.prev_maps[self.current_location[2]][0]) ** 2 + (self.current_location[1] - self.prev_maps[self.current_location[2]][1]) ** 2)
        
        cursor_pos = np.array([-1, -1])
        if self._read_m(0xD057) != 0x00:
            cursor_pos = np.array([self._read_m(0xCC25), self._read_m(0xCC26)]) # cursor x y
        state = np.append(self._simplify_game_area(), cursor_pos)

        return state

    ### REWARD RELATED FUNCTIONS

    def _check_location_reward(self) -> float:
        reward = 0.0

        if self.current_location in self.prev_locations:
            if self.prev_locations[self.current_location] > 4:
                reward = 0
            elif self._read_m(0xD057) == 0x00: 
                if self.current_location[2] != 12: #12 = route 1
                    self.prev_locations[self.current_location] += 1
                reward += 1
        else:
            self.prev_locations[self.current_location] = 1
            if self.current_location[2] not in self.prev_maps:
                reward += 1000
                self.prev_maps[self.current_location[2]] = self.current_location
                if self.current_location[2] == 12: #route 1 
                    reward += 2000
            else:
                reward += 5 + self.distance * 0.2

        self.prev_location = self.current_location
        return reward

    def _calculate_battle_reward(self) -> float:
        game_area = PokemonEnvironment.game_area(self)
        enemy_hp = self._read_m(0xCFE7)
        reward = 10

        if self.enemy_hp and enemy_hp < self.enemy_hp:
            print("ATTACK")
            reward += 1000
        elif np.array_equal(game_area[-7:, :], self.pokemon_select):
            reward -= 20
        elif np.array_equal(game_area[-7:, :], self.battle_flee):
            reward -= 20

        self.enemy_hp = enemy_hp
        return reward

    def _calculate_level_reward(self) -> float:
        if self.prev_gamestate and self.gamestate["xp"] > self.prev_gamestate["xp"]:
            print("DING")
            return 8000
        return 0


    def _calculate_reward(self, new_state: dict) -> float:
        reward = 0.0

        reward += self._calculate_level_reward()
        if self._read_m(0xD057) != 0x00:
            if self.is_in_battle == False:
                reward += 50
            reward += self._calculate_battle_reward()
            self.is_in_battle = True
        elif self.is_in_battle:
            if self.prev_gamestate and self.gamestate["xp"] == self.prev_gamestate["xp"]:
                reward -= 100
            self.is_in_battle = False
        else:
            reward += self._check_location_reward()

        self.prev_gamestate = self.gamestate
        return reward

    ### FINISHED OR TRUNCATED FUNCTIONS

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        if self.steps >= 1000:
            self._reset_episode()
            return True
        return False

    def _reset_episode(self):
        self.prev_gamestate = None
        self.gamestate = None
        self.current_location = None
        self.prev_location = None
        self.enemy_hp = None
        self.is_in_battle = False
        self.distance = 0
        self.prev_locations = {}
        self.prev_maps = {}
