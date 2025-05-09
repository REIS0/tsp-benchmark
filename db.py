import sqlite3


# DATA
# | algorithm | mixer (qaoa) | ansatz (vqe) | optimizer | depth | cost | valid | graph | iteration | optimal | time_sec
class Database:
    def __enter__(self):
        self.con = sqlite3.connect("data.db")
        return self

    def insert_data(self, data: dict):
        cur = self.con.cursor()
        cur.execute(
            """
            INSERT INTO data (algorithm, mixer, ansatz, optimizer, depth, cost, valid, graph, iteration, optimal, time_sec)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.get("algorithm"),
                data.get("mixer"),
                data.get("ansatz"),
                data.get("optimizer"),
                data.get("depth"),
                data.get("cost"),
                data.get("valid"),
                data.get("graph"),
                data.get("iteration"),
                data.get("optimal"),
                data.get("time"),
            ),
        )
        self.con.commit()

    def fetch_data(self) -> list:
        cur = self.con.cursor()
        return cur.execute("SELECT * FROM data").fetchall()

    def run_sql(self, command: str) -> list:
        cur = self.con.cursor()
        return cur.execute(command).fetchall()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.con.close()

    def get_finished_vqe(self) -> list:
        cur = self.con.cursor()
        return cur.execute(
            "SELECT depth, ansatz, optimizer, graph, iteration FROM data WHERE algorithm='vqe'"
        ).fetchall()

    def get_finished_qaoa(self) -> list:
        cur = self.con.cursor()
        return cur.execute(
            "SELECT depth, mixer, optimizer, graph, iteration FROM data WHERE algorithm='qaoa'"
        ).fetchall()
