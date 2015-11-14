import threading

class TrainerThread(threading.Thread):
    def __init__(self, trainer):
        threading.Thread.__init__(self)
        self.trainer = trainer

    def run(self):
        try:
            self.trainer.trainUntilConvergence()
        except IndexError:
            print "Exception in thread"
