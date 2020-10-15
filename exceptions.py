from everest import exceptions

class CrisisModelException(exceptions.EverestException):
    pass

class CrisisModelMissingAsset(exceptions.MissingAsset, CrisisModelException):
    pass
