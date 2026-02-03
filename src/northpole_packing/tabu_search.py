import copy
import random
import time
import numpy as np
from collections import deque  # działa jak FIFO
from northpole_packing.initialization import greedy_initialization
from northpole_packing.const import PRECISION  # ile miejsc po przecinku dla x,y
from northpole_packing.tree import (
    ChristmasTree,
    has_collision_with_candidate,
    calculate_side_length,
    convert_trees_to_string,
)


class TabuSearch:
    def __init__(
        self,
        output_log_path: str,  # plik na potem
        num_trees: int = 100,
        iterations: int = 1000,
        neighbors_per_iter: int = 20,  # ile opcji (sąsiadów) do porównania
        tabu_tenure: int = 150,  # długość pamięci tabu
        step_pos_sigma: float = 0.2,  # o ile najczęściej zmieni się miejsce x albo y (moze być -0.1 a może być 0.7)
        step_angle_sigma: float = 30.0,  # o ile najczęściej zmieni się kąt
        aspiration: bool = True,  # czy mogę złamać tabu (jak widzę genialne rozwiązanie)
    ):
        self.output_log_path = output_log_path
        self.num_trees = num_trees
        self.iterations = iterations
        self.neighbors_per_iter = neighbors_per_iter
        self.tabu_tenure = tabu_tenure
        self.step_pos_sigma = step_pos_sigma
        self.step_angle_sigma = step_angle_sigma
        self.aspiration = aspiration

    # zmieniam: które drzewo, jaki parametr, o ile (opis jednego ruchu - etykieta)
    def _move_signature(
        self, tree_idx: int, param: str, old_val: float, new_val: float
    ):  # które drzewo, jaki parametr, wartość przed i po
        if param in ("x", "y"):
            delta = round(new_val - old_val, PRECISION)  # o ile przesunięte drzewo
        else:
            delta = (new_val - old_val) % 360.0  # jak nie to kąt, kąt w zakresie 0-360
            if delta > 180.0:
                delta -= 360.0
            delta = round(delta, 1)  # zabezpieczenie do floatów i większej precision
        return (tree_idx, param, delta)

    def _generate_neighbor(
        self, trees, tabu_set, best_cost
    ):  # tworzy jednego sąsiada aktualnego rozwiązania jeśli się da, jeśli nie to zwraca (None, None, None, None)

        n = len(trees)
        tree_idx = random.randrange(n)  # losuje jedno drzewo
        tree = trees[tree_idx]
        x, y, angle = tree.get_params()  # biore jego dane paraketry

        param = random.choice(
            ["x", "y", "angle"]
        )  # losowo co będe zmieniać w tym drzewie, tylko 1 rzecz

        if param == "x":
            new_x = round(
                x + float(np.random.normal(0, self.step_pos_sigma)), PRECISION
            )  # przesówanie x, reszta bez zmian, dla innych analogicznie
            new_y = y
            new_angle = angle
            old_val = x
            new_val = new_x
        elif param == "y":
            new_x = x
            new_y = round(
                y + float(np.random.normal(0, self.step_pos_sigma)), PRECISION
            )
            new_angle = angle
            old_val = y
            new_val = new_y
        else:
            new_x = x
            new_y = y
            new_angle = round(
                (angle + float(np.random.normal(0, self.step_angle_sigma))) % 360.0, 1
            )  # zaokrąglanie do 0.1 stopnia
            old_val = angle
            new_val = new_angle

        move_sig = self._move_signature(
            tree_idx, param, old_val, new_val
        )  # to z wyżej, etykieta ruchu to będe ew. wpisywać do tabu np (17, x, -0.15)

        new_tree = ChristmasTree(str(new_x), str(new_y), str(new_angle))

        other_trees = [
            t for i, t in enumerate(trees) if i != tree_idx
        ]  # czy nowe drzewo zderza sie z innymi
        if has_collision_with_candidate(other_trees, new_tree):
            return None, None, None, None

        neighbor_trees = trees[:]  # w kopii listy, podmieniam to jedno drzewo
        neighbor_trees[tree_idx] = new_tree
        neighbor_cost = calculate_side_length(
            neighbor_trees
        )  # jak dobre jest nowe rozwiązanie

        is_tabu = (
            move_sig in tabu_set
        )  # czy ruch był ostatnio robiony(pamięć), tabu_set to pamięć ostatnich ruchów
        if (
            is_tabu and self.aspiration and neighbor_cost < best_cost
        ):  # jeśli nie zmienia najlepszego to nie moge go wykonać
            is_tabu = False  # traktuje go jakby nie był już tabu, ułatwienie na potem

        return (
            neighbor_trees,
            neighbor_cost,
            move_sig,
            is_tabu,
        )  # nowe rozwiązanie, jego koszt, jaki ruch, czy ten ruch tabu

    def solve(self):
        start_time = time.time()
        best_solution = greedy_initialization(
            num_trees=self.num_trees
        )  # greedy na początek jak w SA
        end_time = time.time()
        print(
            f"Initialized starting solution using greedy algorithm: {round(end_time - start_time, 2)} s."
        )
        best_cost = calculate_side_length(best_solution)  # najlepsze jakie mam
        current_solution = copy.deepcopy(
            best_solution
        )  # na start jakie teraz jest rozwiązanie bo potem zmienia się w każdej iteracji
        current_cost = best_cost  # przyjmuje najlepszy początkowy koszyt

        tabu_queue = deque(
            maxlen=self.tabu_tenure
        )  # kolejka FIFO, do pamięci ruchów, Deque - dodawanie i usuwanie elementów z obu końców do (FIFO)/ (LIFO)
        tabu_set = set()  # set szybkie sprawdzanie, set - segregator z ruchami
        max_neighbor_attempts = 20  # na stałe po prostu

        with open(
            self.output_log_path, "w"
        ) as output_log:  # do pliku jak w SA, w każdej iteracji dopisuje linię
            for it in range(
                1, self.iterations + 1
            ):  # max ile iteracji (parametr na początku)
                candidates = []  # dobre ruchy, potem będę wybierać z nich najlepszy

                attempts = 0  # ile razy była próba wpisania sąsiada na listę, niżej
                max_attempts = (
                    self.neighbors_per_iter * max_neighbor_attempts
                )  # neighbors_per_iter = 60 ile sąsiadów porównuję, max_neighbor_attempts razy prubuję dodać na listę następnego sąsiada (jak raz to moze byc kolizja i sie nie doda, a tak mam np 20 razy możliwość dodać tego sąsaiada, po 20 nieudanych razach nie dodaje sąsiada i potem porównuję np listę z 59 sąsiadami)

                while (
                    len(candidates) < self.neighbors_per_iter
                    and attempts < max_attempts
                ):  # Losowanie sąsiadów aż jest wystarczająco albo skończą się próby
                    attempts += 1
                    neigh, cost, move_sig, is_tabu = self._generate_neighbor(
                        current_solution, tabu_set, best_cost
                    )
                    if (
                        neigh is None
                    ):  # Jeśli _generate_neighbor zwrócił (None, None, None, None), to próbuje dalej jeśli jeszcze mogę
                        continue
                    if is_tabu:  # Jeśli ruch jest tabu, ignoruje próbuje dalej
                        continue
                    candidates.append((cost, neigh, move_sig))  # mamy kandydata

                if not candidates:

                    best_solution_str = convert_trees_to_string(
                        best_solution
                    )  # jeśli w iteracji nie znajdę żadnego z rozwiązan, w zadnej próbie, kończe program
                    output_log.write(
                        f"{it};NO_MOVE;{best_cost};{current_cost};{best_solution_str}\n"
                    )
                    print(
                        f"[TabuSearch] Zatrzymano w iteracji {it}: "
                        f"brak legalnych ruchów, zmień parametry "
                    )
                    break

                neighbor_cost, neighbor_sol, move_sig = min(
                    candidates, key=lambda x: x[0]
                )  # Z zebranych kandydatów wybieram po prostu ten z najmniejszym kosztem

                current_solution = neighbor_sol  # aktualizacja rozwiązania
                current_cost = neighbor_cost

                if (
                    current_cost < best_cost
                ):  # nowe aktualne jest lepsze niż najlepsze dotąd, to aktualizuje best
                    best_solution = copy.deepcopy(current_solution)
                    best_cost = current_cost

                if (
                    tabu_queue.maxlen is not None
                    and len(tabu_queue) == tabu_queue.maxlen
                ):  # Jak tabu jest pełne to najstarszy ruch, czyli znowu nie jest traktowany jak tabu, żeby każdy ruch nie był tabu, to niektóre zapomina
                    removed = tabu_queue.popleft()
                    tabu_set.discard(removed)

                tabu_queue.append(move_sig)  # ale nowy zapisz ruch
                tabu_set.add(move_sig)  # do set też

                best_solution_str = convert_trees_to_string(
                    best_solution
                )  # do zapisu w logu
                output_log.write(
                    f"{it};{self.tabu_tenure};{best_cost};{current_cost};{best_solution_str}\n"
                )
                output_log.flush()

        return best_solution, best_cost  # cały układ drtzew
