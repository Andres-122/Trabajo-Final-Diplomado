import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

app = Flask(__name__)

file_path = "https://raw.githubusercontent.com/Andres-122/Ejemplogit/master/db.csv"
# Leer el archivo CSV usando pandas
df = pd.read_csv(file_path, delimiter=';')
# Limpiar datos
df = df.dropna()    

# Convertir la columna 'FECHA_NACIMIENTO' a tipo datetime
df['FECHA_NACIMIENTO'] = pd.to_datetime(df['FECHA_NACIMIENTO'])
# Calcular la edad restando la fecha actual de la fecha de nacimiento y dividiendo por el número de días en un año
df['EDAD'] = ((pd.to_datetime('2019-12-31') - df['FECHA_NACIMIENTO']).dt.days / 365.25).astype(int)

# Preparar los datos
X = df[['EDAD', 'SEXO', 'ESTRATO', 'JORNADA', 'MODALIDAD', 'NOMBRE_SEDE', 'TIPO_IDEN_EST', 'NOMBRE_FACULTAD', 'LUGAR_NACIMIENTO',]]
y = df['NOMBRE_ESTADO']

# Aplicar codificación one-hot
encoder = OneHotEncoder(drop='first')
X_encoded = encoder.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Crea un clasificador SVM con kernel lineal y los hiperparámetros especificados
svm_classifier = SVC(kernel='linear', C=10, gamma=0.01)

# Aplica validación cruzada con 5 divisiones (folds)
scores = cross_val_score(svm_classifier, X_train, y_train, cv=5)

# Entrena el clasificador SVM con todos los datos de entrenamiento
svm_classifier.fit(X_train, y_train)

# Obtener las categorías únicas para cada característica categórica
categorias_sexo = X['SEXO'].unique().tolist()
categorias_jornada = X['JORNADA'].unique().tolist()
categorias_modalidad = X['MODALIDAD'].unique().tolist()
categorias_sede = X['NOMBRE_SEDE'].unique().tolist()
categorias_tipo_iden = X['TIPO_IDEN_EST'].unique().tolist()
categorias_nombre_facultad = X['NOMBRE_FACULTAD'].unique().tolist()
categorias_lugar_nacimiento = X['LUGAR_NACIMIENTO'].unique().tolist()

# Ruta para la página principal
@app.route('/')
def home():
    return render_template('index2.html', categorias_sexo=categorias_sexo,
                           categorias_jornada=categorias_jornada,
                           categorias_modalidad=categorias_modalidad,
                           categorias_sede=categorias_sede,
                           categorias_tipo_iden=categorias_tipo_iden,
                           categorias_nombre_facultad=categorias_nombre_facultad,
                           categorias_lugar_nacimiento=categorias_lugar_nacimiento,
                           accuracy=scores.mean()) 

# Ruta para la predicción
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Obtener los datos del formulario
        edad = float(request.form['EDAD'])
        sexo = request.form['SEXO']
        estrato = float(request.form['ESTRATO'])
        jornada = request.form['JORNADA']
        modalidad = request.form['MODALIDAD']
        nombre_sede = request.form['NOMBRE_SEDE']
        tipo_iden_est = request.form['TIPO_IDEN_EST']
        nombre_facultad = request.form['NOMBRE_FACULTAD']
        lugar_nacimiento = request.form['LUGAR_NACIMIENTO']

        # Crear un DataFrame con los nuevos datos
        new_data = pd.DataFrame({'EDAD': [edad], 'SEXO': [sexo], 'ESTRATO': [estrato],
                                 'JORNADA': [jornada], 'MODALIDAD': [modalidad],
                                 'NOMBRE_SEDE': [nombre_sede], 'TIPO_IDEN_EST': [tipo_iden_est],
                                 'NOMBRE_FACULTAD': [nombre_facultad],'LUGAR_NACIMIENTO': [lugar_nacimiento]})

        # Aplicar codificación one-hot
        new_data_encoded = encoder.transform(new_data)

        # Realizar la predicción con el clasificador SVM entrenado
        prediction = svm_classifier.predict(new_data_encoded)[0]

        # Renderizar la plantilla de resultados
        return render_template('result2.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
