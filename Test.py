from Pricer import Pricer
from option import AmericanPutOption

if __name__ == "__main__":
    option = AmericanPutOption(1, 1)
    pricer = Pricer(1, 0.25, rate=0.06, vol=0.2, path_size=100000)
    print(pricer(option))
