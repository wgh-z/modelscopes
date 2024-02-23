# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks


class Model():
    def __init__(self):
        pass
    # def __init__(self, model_id):
        # self.model = pipeline(Tasks.domain_specific_object_detection, model=model_id)

    def __call__(self) -> None:
        self.results = Results()
        while True:
            yield self.results
            self.results.box = self.results.box

    # def __iter__(self) -> None:
    #     while True:
    #         yield 'iter'

    # def __next__(self) -> None:
    #     print('next')

    def __len__(self) -> None:
        pass


class Results:
    def __init__(self):
        self.name = {'0': 'safety hat', '1': 'no safety hat'}
        self.box = [0, 0, 0, 0]
    

    # def __getitem__(self, idx):
    #     """Return a Results object for the specified index."""
    #     r = self.new()
    #     for k in self.keys:
    #         setattr(r, k, getattr(self, k)[idx])
    #     return r

    # def new(self):
    #     """Return a new Results object with the same image, path, and names."""
    #     return Results()

    # @property
    # def keys(self):
    #     """Return a list of non-empty attribute names."""
    #     return [k for k in self._keys if getattr(self, k) is not None]

model = Model()
results = model()
while True:
    result = next(results)
    print(result.name)