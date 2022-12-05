# book-recommendation-system_using_h2o-wave



https://user-images.githubusercontent.com/67860730/205605228-cf2eaa5e-e69b-4eae-98aa-c913e3d28c42.mp4


This is a simple [collaborative filtering](https://developers.google.com/machine-learning/recommendation/collaborative/basics) based book recommendation system made using [h2o wave](https://wave.h2o.ai/).

## Getting Started

### System Requirements

1. Python 3.7+
2. pip3

### 1. Clone the repository:

``` bash
git clone https://github.com/ChathurindaRanasinghe/book-recommendation-system_using_h2o-wave.git
```

### 2. Create a virtual environment:

``` bash
python3 -m venv venv
```

### 3. Activate the virtual environment:
``` bash
source venv/bin/activate
```

**windows**
``` bash
venv\Scripts\activate.bat
```
To deactivate the virtual environment use ```deactivate``` command.

### 4. Install dependencies:

``` bash
(venv) pip3 install -r requirements.txt 
```

### 5. Run the app:
``` bash
(venv) wave run app
```

### 6. View the app:
Point your favorite web browser to http://localhost:10101/recommender

## Testing

Tests are located in ```/tests```. See instructions on [h2o wave docs](https://wave.h2o.ai/docs/browser-testing) to configure tests.

## Others

* [h2o wave documentation](https://wave.h2o.ai/docs/getting-started)





