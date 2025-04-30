import os
import re
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    print("Tavily api key is not set.")
else:
    #tool = TavilySearch(max_results=5, topic="general", search_depth="advanced")
    #result = tool.invoke({"query": "Where is Kecskemét? What should I know about it?"})
    data = {'query': 'Where is Kecskemét? What should I know about it?', 'follow_up_questions': None, 'answer': None, 'images': [], 'results': [{'title': 'Kecskemét - Wikipedia', 'url': 'https://en.wikipedia.org/wiki/Kecskemét', 'content': "Kecskemét (US: / ˈ k ɛ tʃ k ɛ m eɪ t / KETCH-kem-ayt [3] [4] Hungarian: [ˈkɛt͡ʃkɛmeːt]) is a city with county rights in central Hungary.It is the eighth-largestt city in the country, and the county seat of Bács-Kiskun.. Kecskemét lies halfway between the capital Budapest and the country's third-largest city, Szeged, 86 kilometres (53 miles) from both of them and almost equal", 'score': 0.9016528, 'raw_content': None}, {'title': 'Kecskemét - everyone should visit the city of tolerance at least once', 'url': 'https://info-budapest.com/kecskemet/', 'content': 'Kecskemét, the centre of Hungary. Kecskemét lays at the heart of Hungary, between the rivers Danube and Tisza, south of Budapest. The ancient core of the town was created by the intersection of trade routes. Kecskemét, using its favourable locations quickly emerged from the surrounding towns as a tax collection and sales point.', 'score': 0.8757978, 'raw_content': None}, {'title': 'Visit Kecskemet, Hungary - the Art Nouveau Paradise - Kami and the ...', 'url': 'https://www.mywanderlust.pl/visit-kecskemet-hungary/', 'content': 'Kecskemet became an important center of agriculture in the region, which led to the enrichment of the city. Still today, the place is known as the Hungarian capital of barackpalinka, a local apricot brandy. With the growth of wealth, Kecskemet invested in its looks, too, and hence a large number of amazing art nouveau buildings in the city.', 'score': 0.6164054, 'raw_content': None}, {'title': 'Kecskemét - Travel guide at Wikivoyage', 'url': 'https://en.wikivoyage.org/wiki/Kecskemét', 'content': "Here is the 'Kecskemét purgatory relief' A sculpture composition at the Kecskemét Franciscan church at Kossuth square fence wall. Beneath the cross, purgatory relief depicting made in 1792 and it was a remnant of Calvary. The crucifix, Gábor Imre sculptor in 1943 was stayyed above the relief. ... Pretty much everything you need to know is in", 'score': 0.49831602, 'raw_content': None}, {'title': 'Kecskemet, Hungary: All You Must Know Before You Go (2025 ... - Tripadvisor', 'url': 'https://www.tripadvisor.com/Tourism-g274896-Kecskemet_Bacs_Kiskun_County_Southern_Great_Plain-Vacations.html', 'content': "If you're a more budget-conscious traveler, then you may want to consider traveling to Kecskemet between March and May, when hotel prices are generally the lowest. Peak hotel prices generally start between September and November. ©", 'score': 0.49699774, 'raw_content': None}], 'response_time': 1.57}
    results = data["results"]
    
    def find_largest_score_index(data: list):
        max = 0
        for i in range(0, len(data)):
            if data[i]["score"] > data[max]["score"]:
                max = i
        return max
    
    largest_score_index = find_largest_score_index(results)
    print(results[largest_score_index]["content"])