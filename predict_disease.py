from gradio_client import Client, handle_file
class prediction_disease_type:
    def __init__(self):
        self.HEIGHT = 256
        self.WIDTH = 256
        self.client = Client("bhanusAI/plantifysol")

        print("done")
    def get_label(self,img,plant_type):
        print(img)
        result = self.client.predict(
		    img=handle_file(img),plant_type=plant_type,
		    api_name="/predict"
        )
        print(result)
        return result
     

if __name__ == "__main__":
    pass
        
