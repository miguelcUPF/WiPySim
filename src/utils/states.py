class MACState:
    IDLE = 0
    CONTEND = 1
    BACKOFF = 2
    TX = 3
    WAIT_FOR_BACK = 4
    RX = 5