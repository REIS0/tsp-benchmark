import sqlite3


# DATA
# | algorithm | mixer (qaoa) | ansatz (vqe) | optimizer | depth | feasibility_ratio | cost_ratio | best_ratio
class Database:
    def __enter__(self):
        self.con = sqlite3.connect('data.db')
        return self

    def insert_data(self, data: dict):
        cur = self.con.cursor()
        cur.execute(
            """
            INSERT INTO data (algorithm, mixer, ansatz, optimizer, depth, feasibility_ratio, cost_ratio, rank, time_sec) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.get('algorithm'),
                data.get('mixer'),
                data.get('ansatz'),
                data.get('optimizer'),
                data.get('depth'),
                data.get('feasibility_ratio'),
                data.get('cost_ratio'),
                data.get('rank'),
                data.get('time'),
            )
        )
        self.con.commit()

    def fetch_data(self) -> list:
        cur = self.con.cursor()
        return cur.execute("SELECT * FROM data").fetchall()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.con.close()

    def get_finished_vqe(self) -> list:
        cur = self.con.cursor()
        return cur.execute("SELECT depth, ansatz, optimizer FROM data").fetchall()

    def get_finished_qaoa(self) -> list:
        cur = self.con.cursor()
        return cur.execute("SELECT depth, mixer, optimizer FROM data").fetchall()
