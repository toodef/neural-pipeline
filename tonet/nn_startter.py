from neural_pipeline.tonet.neuro_studio.PySide2Wrapper.PySide2Wrapper import Application
from neural_pipeline.tonet.neuro_studio.nn_studio import NeuralStudio

if __name__ == "__main__":
    app = Application()
    studio = NeuralStudio()
    resolution = app.screen_resolution()
    studio.resize(1000, 0)
    studio.move(100, 100)
    studio.show()
    app.run()
