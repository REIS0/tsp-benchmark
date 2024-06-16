from qiskit_ibm_runtime import QiskitRuntimeService


def get_key() -> QiskitRuntimeService:
    file = open("key", "r")
    key = str(file.read())

    service = QiskitRuntimeService(
        channel="ibm_quantum",
        token=key
    )

    file.close()

    return service
