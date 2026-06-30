class Benchmark:
    
    def __init__(self, train_stream, test_stream, valid_stream=None,blurred_stream=None):
        
        self.train_stream = train_stream
        self.test_stream = test_stream
        self.valid_stream = valid_stream
        self.blurred_stream = blurred_stream
