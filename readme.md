# Recipe Generation API

This project provides a REST API that generates recipes based on a given list of ingredients. It uses a T5 transformer model fine-tuned on a recipe dataset, provided by Hugging Face. 

## How it Works

The API is built with [FastAPI](https://fastapi.tiangolo.com/), a modern, fast (high-performance) web framework for building APIs with Python 3.6+ based on standard Python type hints.

The API has one main endpoint, `/generate_recipes`, which accepts a list of ingredients as input and returns a list of generated recipes. The endpoint uses the T5 transformer model to generate the recipe texts, which are then parsed into recipe objects by the `parse_generated_recipes` function. 

Here is a brief explanation of some of the functions used in this project:

```python
def generation_function(texts):
    _inputs = texts if isinstance(texts, list) else [texts]
    inputs = [prefix + inp for inp in _inputs]
    inputs = tokenizer(
        inputs, 
        max_length=256, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt",
    )

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    output_ids = model.generate(
        input_ids=input_ids, 
        attention_mask=attention_mask,
        **generation_kwargs
    )
    generated_recipe = target_postprocessing(
        tokenizer.batch_decode(output_ids, skip_special_tokens=False),
        special_tokens
    )
    return generated_recipe
```

This function is responsible for generating the recipe text based on the given input ingredients. It uses the pre-trained T5 transformer model from Hugging Face to generate the text.

```python
def skip_special_tokens(text, special_tokens):
    for token in special_tokens:
        text = text.replace(token, "")

    return text
```

This function removes special tokens from the generated recipe text.

```python
def parse_generated_recipes(recipe_list):
    parsed_recipes = []
    for recipe in recipe_list:
        recipe_obj = {}
        title_match = re.search(r" title: (.+?)\n", recipe)
        ingredients_match = re.search(r" ingredients: (.+?)\n directions:", recipe)
        directions_match = re.search(r" directions: (.+)", recipe)

        if title_match:
            recipe_obj["title"] = title_match.group(1).strip()
        if ingredients_match:
            ingredients = ingredients_match.group(1).strip().split('--')
            recipe_obj["ingredients"] = [ingredient.strip() for ingredient in ingredients]
        if directions_match:
            directions = directions_match.group(1).strip().split('--')
            recipe_obj["directions"] = [direction.strip() for direction in directions]

        parsed_recipes.append(recipe_obj)
    return parsed_recipes
```

This function parses the generated recipe text into a recipe object that can be returned by the API.

## How to Run

To run the API, first make sure you have Docker installed on your machine. Then, open a terminal and navigate to the project directory. Build the Docker image by running the following command:

```bash
docker build -t recipe_generation_api .
```

This will build the Docker image with the tag `recipe_generation_api`. Once the image is built, run the following command to start the container:

```bash
docker run -p 8000:8000 -d recipe_generation_api
```

This will start the container in detached mode (`-d`) and map the container port 8000 to the host port 8000 (`-p 8000:8000`). The API will