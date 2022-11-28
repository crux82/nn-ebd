from multi_exec import multi_config_exec


def main():
    """
    multi_config_exec(SEED) run code with seed "SEED" to get reproducible results with PyTroch
    """
    multi_config_exec(9647566)
    # multi_config_exec(34204329)
    # multi_config_exec(7763456)
    # multi_config_exec(400423)
    # multi_config_exec(92578432)
    #...you can add as many runs as you like

if __name__ == "__main__":
    main()



