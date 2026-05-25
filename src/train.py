from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from preprocess import clean_data, load_data, prepare_features


def train_model():
	data = load_data()
	data_clean = clean_data(data)
	x, y = prepare_features(data_clean)

	x_train, x_test, y_train, y_test = train_test_split(
		x,
		y,
		test_size=0.2,
		random_state=42,
		stratify=y,
	)

	model = LogisticRegression(max_iter=2000, solver="liblinear")
	model.fit(x_train, y_train)

	y_pred = model.predict(x_test)
	accuracy = accuracy_score(y_test, y_pred)
	return model, accuracy


def main():
	_, accuracy = train_model()
	print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
	main()
