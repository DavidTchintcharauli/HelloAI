let numberModel;
let useModel;

window.addEventListener("load", async () => {
  try {
    await loadModels();
    document.getElementById("chatButton").disabled = false;

    // Enter ღილაკზე რეაგირება
    document.getElementById("chatInput").addEventListener("keypress", function(event) {
      if (event.key === "Enter") {
        handleUserInput();
      }
    });

  } catch (err) {
    console.error("მოდელების ჩატვირთვის შეცდომა:", err);
  }
});

async function loadModels() {
  useModel = await use.load();
  numberModel = tf.sequential();
  numberModel.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  numberModel.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

  const xs = tf.tensor2d([[1], [2], [3], [4], [5]], [5, 1]);
  const ys = tf.tensor2d([[2], [4], [6], [8], [10]], [5, 1]);

  await numberModel.fit(xs, ys, { epochs: 1000 });
}

function cosineSimilarity(a, b) {
  const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dot / (normA * normB);
}

async function handleUserInput() {
  const inputText = document.getElementById("chatInput").value.trim();
  if (!inputText) {
    document.getElementById("chatResponse").innerText = "შეიყვანე ტექსტი!";
    return;
  }

  const parsedNumber = parseFloat(inputText);
  if (!isNaN(parsedNumber)) {
    if (!numberModel) {
      document.getElementById("chatResponse").innerText = "რიცხვების მოდელი ჯერ არ არის მზად...";
      return;
    }
    const inputTensor = tf.tensor2d([[parsedNumber]], [1, 1]);
    const prediction = numberModel.predict(inputTensor);
    const result = await prediction.data();
    document.getElementById("chatResponse").innerText = `გამრავლებული შედეგი: ${result[0].toFixed(2)}`;
    return;
  }

  if (!useModel) {
    document.getElementById("chatResponse").innerText = "ტექსტის მოდელი ჯერ არ არის მზად...";
    return;
  }

  const knownGreetings = ["გამარჯობა", "სალამი", "ჰეი", "დილა მშვიდობისა", "მოგესალმები"];
  const sentences = [inputText, ...knownGreetings];
  const embeddings = await useModel.embed(sentences);
  const vectors = await embeddings.array();

  const inputVector = vectors[0];
  let maxSim = 0;

  for (let i = 1; i < vectors.length; i++) {
    const sim = cosineSimilarity(inputVector, vectors[i]);
    if (sim > maxSim) maxSim = sim;
  }

  if (maxSim > 0.7) {
    const variants = [
      "მიხარია შენი ნახვა!",
      "როგორ ხარ დღეს?",
      "კეთილი იყოს შენი მობრძანება!",
      "გისურვებ კარგ დღეს!",
      "მითხარი, როგორ დაგეხმარო?",
      "დღეს მშვენიერი ამინდია!",
      "მომენატრე! რას შვები?",
      "მზად ვარ, მომიყევი რამე ახალი!"
    ];
    const randomResponse = variants[Math.floor(Math.random() * variants.length)];
    document.getElementById("chatResponse").innerText = randomResponse;
  } else {
    document.getElementById("chatResponse").innerText = "ბოდიში, ვერ გაგიგე 😢";
  }
}
