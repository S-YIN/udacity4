import os
import pandas as pd
from pydantic import BaseModel
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from dotenv import load_dotenv

load_dotenv()  
# os.environ['OPENAI_API_KEY'] = "" #your API Key


#### Step 2 Generating Real Estate Listings ####
class house(BaseModel):
    Neighborhood: str = Field(description="The name of the house neighborhood.")
    Price: str = Field(description="The price of the house")
    Bedrooms: str = Field(description="The number of bedrooms")
    Bathrooms: str = Field(description="The number of bathrooms")
    HouseSize: str = Field(description="The size of the house in sqft")
    Description: str = Field(description="Realtor description of the house. Include some property amenities details such as backyard, garage")
    NeighborhoodDescription: str = Field(description="Realtor description of the neighborhood. Include the transportation, school, urban/suburban, city, etc.")
    
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
structured_llm = model.with_structured_output(house)

houses = []
for i in range(3):
    house = structured_llm.invoke(
        "Randomly generate a realtor post of a house in the market. Response in JSON format"
    )
    houses.append(house)

for i in range(3):
    house = structured_llm.invoke(
        "Randomly generate a realtor post of a budget friendly house in the market. Response in JSON format"
    )
    houses.append(house)

for i in range(3):
    house = structured_llm.invoke(
        "Randomly generate a realtor post of a luxury house in the market. Response in JSON format"
    )
    houses.append(house)

for i in range(3):
    house = structured_llm.invoke(
        "Randomly generate a realtor post of a house in a suburban city. Response in JSON format"
    )
    houses.append(house)

df = pd.DataFrame([vars(s) for s in houses])
df.to_csv("listings.csv", index=False)


#### Step 3: Storing Listings in a Vector Database ####
## create chroma db
embedding_function = SentenceTransformerEmbeddings(model_name="thenlper/gte-base")

loader = CSVLoader("./listings.csv")
documents = loader.load()
db = Chroma.from_documents(documents, embedding_function, persist_directory="./chroma_db")

# load from disk
db2 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)


#### Step 4: Building the User Preference Interface ####
## User Preference
questions = [   
                "How big do you want your house to be?" 
                "What are 3 most important things for you in choosing this property?", 
                "Which amenities would you like?", 
                "Which transportation options are important to you?",
                "How urban do you want your neighborhood to be?",   
]
answers = [
    "A comfortable three-bedroom house with a spacious kitchen and a cozy living room.",
    "A quiet neighborhood, good local schools, and convenient shopping options.",
    "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
    "Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads.",
    "A balance between suburban tranquility and access to urban amenities like restaurants and theaters."
]

preferences = " ".join(answers)


#### Step 5: Searching Based on Preferences ####
docs = db2.similarity_search(preferences)
print("Top 3 properties based on the user preferences:")
print(docs[0].page_content)
print(docs[1].page_content)
print(docs[2].page_content)



#### Step 6: Personalizing Listing Descriptions ####
def personal_description(listing, preferences):
    prompt_template = PromptTemplate.from_template(
        """You are an assistant to generate tailored description and recommandation to users.
    Tailoring the description from property listing to resonate with the buyer’s specific preferences. 
    This involves subtly emphasizing aspects of the property that align with what the buyer is looking for.
    Ensure that the augmentation process enhances the appeal of the listing without altering factual information.
    Write the description in a single paragraph.

    Property listing: {listing}

    Buyer preference: {preference}
    """
    )
    llm_chain = LLMChain(
        prompt=prompt_template,
        llm=model
    )
    output = llm_chain({"listing": listing, "preference": preferences})
    return output

for i in range(3):
    output = personal_description(docs[i].page_content, preferences)
    print(i,". ",output["text"],"\n\n")



#### Testing and example outputs ####
# example user prefereces
preferences = "A comfortable three-bedroom house with a spacious kitchen and a cozy living room. A quiet neighborhood, good local schools, and convenient shopping options. A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system. Easy access to a reliable bus line, proximity to a major highway, and bike-friendly roads. A balance between suburban tranquility and access to urban amenities like restaurants and theaters."
db2 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
docs = db2.similarity_search(preferences)
print("Top 3 properties based on the user preferences:")
print(docs[0].page_content)
print(docs[1].page_content)
print(docs[2].page_content)
# top 3 matchings
############################################################
# Neighborhood: Sunnyvale
# Price: $250,000
# Bedrooms: 3
# Bathrooms: 2
# HouseSize: 1400
# Description: This charming 3-bedroom, 2-bathroom home features a spacious backyard perfect for family gatherings, an attached garage, and a cozy fireplace in the living room. The kitchen is equipped with modern appliances and plenty of counter space for cooking enthusiasts.
# NeighborhoodDescription: Sunnyvale is a family-friendly suburban neighborhood with excellent schools, parks, and convenient access to public transportation. Enjoy a peaceful environment while being just minutes away from shopping centers and recreational activities.
############################################################
# Neighborhood: Maplewood
# Price: $450,000
# Bedrooms: 4
# Bathrooms: 3
# HouseSize: 2,500
# Description: This stunning 4-bedroom, 3-bathroom home features a spacious open floor plan, a modern kitchen with stainless steel appliances, and a cozy fireplace in the living room. Enjoy the large backyard perfect for entertaining, along with a two-car garage and energy-efficient windows.
# NeighborhoodDescription: Maplewood is a vibrant suburban neighborhood known for its excellent schools and family-friendly atmosphere. With easy access to public transportation and local parks, residents enjoy a mix of urban convenience and suburban tranquility.
############################################################
# Neighborhood: Maplewood
# Price: $250,000
# Bedrooms: 3
# Bathrooms: 2
# HouseSize: 1500
# Description: Charming 3-bedroom, 2-bath home featuring a spacious backyard, a cozy fireplace, and a modern kitchen with updated appliances. Perfect for family gatherings and outdoor activities!
# NeighborhoodDescription: Maplewood is a friendly suburban neighborhood with excellent schools, parks, and easy access to public transportation. Enjoy a peaceful community atmosphere while being just a short drive away from downtown amenities.


# generate Personalizing Listing Descriptions for each of the three properties
for i in range(3):
    output = personal_description(docs[i].page_content, preferences)
    print(i+1,". ",output["text"],"\n\n")

############################################################
# 1 .  Welcome to your dream home in the serene neighborhood of Sunnyvale, 
# where comfort meets convenience! This charming 3-bedroom, 2-bathroom residence is designed 
# for those who appreciate a spacious kitchen, perfect for culinary adventures, 
# and a cozy living room featuring a warm fireplace, ideal for relaxing evenings. 
# The expansive backyard invites you to cultivate your gardening passion or host family 
# gatherings, while the attached two-car garage provides ample storage and parking. 
# With modern appliances and a focus on energy efficiency, this home ensures both 
# functionality and sustainability. Nestled in a family-friendly community with excellent 
# chools, parks, and easy access to public transportation, you’ll enjoy the 
# tranquility of suburban living while being just minutes away from shopping centers, 
# restaurants, and recreational activities. Plus, with bike-friendly roads and proximity 
# to a major highway, commuting and exploring the area is a breeze. Don’t miss the 
# opportunity to make this delightful home your own! 
############################################################
# 2 .  Welcome to your dream home in the vibrant Maplewood neighborhood, where comfort 
# meets convenience! This beautifully designed 4-bedroom, 3-bathroom residence offers a 
# spacious open floor plan that is perfect for family gatherings and entertaining. 
# The modern kitchen, equipped with stainless steel appliances, is a chef's delight, 
# while the cozy fireplace in the living room creates a warm and inviting atmosphere for 
# relaxation. Step outside to discover a large backyard, ideal for gardening enthusiasts 
# and outdoor activities, complemented by a two-car garage for your convenience. 
# With energy-efficient windows and a modern heating system, this home ensures both comfort 
# and sustainability. Nestled in a family-friendly community known for its excellent schools, 
# you'll enjoy easy access to public transportation, local parks, and a variety of shopping 
# options, all while relishing the balance of suburban tranquility and urban amenities 
# like restaurants and theaters. Don’t miss the opportunity to make this exceptional 
# property your new home! 
############################################################
# 3 .  Welcome to your dream home in the charming Maplewood neighborhood, where comfort 
# meets convenience! This delightful 3-bedroom, 2-bath residence is perfect for 
# families seeking a peaceful retreat while enjoying easy access to urban amenities. 
# The spacious kitchen, equipped with modern appliances, is ideal for culinary 
# enthusiasts, while the cozy living room, complete with a warm fireplace, invites 
# relaxation and family gatherings. Step outside to discover a generous backyard, 
# perfect for gardening or outdoor activities, and a two-car garage for added convenience. 
# With excellent local schools, nearby parks, and bike-friendly roads, you’ll appreciate 
# the balance of suburban tranquility and vibrant community life. Plus, with reliable 
# public transportation and quick access to major highways, your commute to downtown 
# restaurants and theaters is a breeze. Don’t miss the opportunity to make this 
# charming home your own! 