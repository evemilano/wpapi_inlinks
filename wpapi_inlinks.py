# %% 
### .\amb\Scripts\activate ###

# Importazione delle librerie di sistema e di rete
import os  # Per la gestione di file e directory
import requests  # Per fare richieste HTTP
import json  # Per lavorare con dati JSON
from urllib.parse import urljoin  # Per costruire URL assoluti

# Importazione per il parsing e la manipolazione di HTML
from bs4 import BeautifulSoup  # Per il parsing di HTML e XML
from html import unescape  # Per decodificare caratteri HTML

# Importazione per il preprocessing del testo
import re  # Per la manipolazione di stringhe tramite espressioni regolari
import nltk  # Per il Natural Language Processing
from nltk.corpus import stopwords  # Per le stopword
from nltk.tokenize import word_tokenize  # Per la tokenizzazione
nltk.download('punkt')  # Download dei dati per la tokenizzazione
nltk.download('stopwords')  # Download delle stopword

# Importazione di SpaCy per il NLP avanzato
import spacy  # Per il Natural Language Processing
nlp = spacy.load("it_core_news_lg", disable=["parser", "ner", "tok2vec"])  # Caricamento del modello italiano

# Importazione per il clustering e l'analisi dei topic
from bertopic import BERTopic  # Per la modellazione dei topic con BERTopic
from sentence_transformers import SentenceTransformer  # Per la rappresentazione vettoriale delle frasi

# Importazione per la visualizzazione dei dati
import plotly.graph_objects as go  # Per creare grafici interattivi

# Importazione per il web scraping e la navigazione
import webbrowser  # Per aprire URL nei browser

# Importazione per l'analisi delle reti
import networkx as nx  # Per creare e analizzare grafi e reti

# Importazione per la gestione e l'analisi dei dati
import pandas as pd  # Per la manipolazione di dati tabellari


# %%
domain = 'www.evemilano.com' # senza https
# sanitari.online
# www.aerografartitalia.it

# %%
def verify_url(base_url, endpoint):
    # Costruisce l'URL finale e verifica che non ci siano doppi slash
    final_url = urljoin(base_url, endpoint)
    
    # Esegue la richiesta HTTP per ottenere lo stato del sito
    try:
        response = requests.get(final_url)
        if response.status_code == 200:
            print(f"URL verificato: {final_url}")
            return True, final_url
            
        else:
            print(f"URL non raggiungibile: {final_url}")
            return False, final_url
            
    except requests.exceptions.RequestException as e:
        print(f"Errore durante la richiesta: {e}")
        return False, final_url

# %%
# Verifica che l'URL del dominio inizi con "http://" o "https://"

# rimuovi eventuale / finale
# Definizione della funzione per rimuovere lo slash finale dall'URL
def remove_trailing_slash(url):
    return url.rstrip('/')

# Esempio di utilizzo della funzione
domain = remove_trailing_slash(domain)

# Se non inizia con uno di questi, "https://" viene aggiunto come prefisso

# Imposta gli URL di base per le API dei post e delle pagine
post_url = f'{domain}/wp-json/wp/v2/posts'
page_url = f'{domain}/wp-json/wp/v2/pages'

# Verifica e aggiusta l'URL per i post se necessario
if not post_url.startswith(('http://', 'https://')):
    post_url = 'https://' + post_url

# Verifica e aggiusta l'URL per le pagine se necessario
if not page_url.startswith(('http://', 'https://')):
    page_url = 'https://' + page_url

# Verifica e aggiusta l'URL del dominio se necessario
if not domain.startswith(('http://', 'https://')):
    domain = 'https://' + domain

# Lista degli endpoint API che vogliamo verificare
endpoints = [
    '/wp-json/wp/v2/posts',
    '/wp-json/wp/v2/pages'
]

# Verifica la validità di ciascun endpoint
for endpoint in endpoints:
    # La funzione verify_url dovrebbe restituire lo stato della verifica e l'URL finale
    status, final_url = verify_url(domain, endpoint)
    
    # Stampa il risultato della verifica
    if status:
        print(f"L'URL {final_url} è valido e restituisce uno status code 200.")
    else:
        print(f"L'URL {final_url} non è valido o non restituisce uno status code 200.")

# %%
# download content
# Inizializzazione di una lista vuota per contenere tutti i contenuti scaricati
all_contents = []

# Definizione della funzione per scaricare contenuti (post o pagine)
def download_contents(url, content_list):
    # Inizializzazione della variabile della pagina
    page = 1
    
    # Loop infinito per continuare a scaricare pagine fino a quando non ce ne sono più
    while True:
        # Eseguire una richiesta GET alla pagina corrente
        #response = requests.get(url, params={'page': page, 'per_page': 100})
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(url, params={'page': page, 'per_page': 100}, headers=headers)

        print(f'Checking {url} page {page}')
        
        # Verifica se la richiesta è stata un successo (status code 200)
        if response.status_code == 200:
            # Estrae i contenuti dalla risposta come JSON e li converte in un oggetto Python
            contents = json.loads(response.text)
            
            # Interrompe il ciclo se la pagina non contiene più contenuti
            if not contents:
                break
                
            # Aggiunge i contenuti appena scaricati alla lista di contenuti esistente
            content_list.extend(contents)
            
            # Passa alla pagina successiva
            page += 1
        else:
            # Stampa un messaggio di errore e interrompe il ciclo se la richiesta fallisce
            print(f"Failed to get data for page {page}: {response.status_code}")
            break

# Scaricare tutti i post utilizzando l'URL specificato per i post
print('Downloading posts...')
download_contents(post_url, all_contents)
print(f"Downloaded {len(all_contents)} posts.")

# Scaricare tutte le pagine utilizzando l'URL specificato per le pagine
print('Downloading pages...')
download_contents(page_url, all_contents)
print(f"Downloaded {len(all_contents)} posts and pages in total.")

# %%
# pulizia content da stopword e tag
print('Removing stopwords and cleaning tags')

# Stopwords combinate
nltk_stopwords = set(stopwords.words('italian'))
spacy_stopwords = nlp.Defaults.stop_words
combined_stopwords = nltk_stopwords.union(spacy_stopwords)

# Funzione di pulizia unificata
def clean_combined_text(title, raw_html):
    # Decodifica entità HTML nel titolo e nel contenuto
    title = unescape(title)
    soup = BeautifulSoup(raw_html, "html.parser")
    content = soup.get_text()

    # Concatena titolo e contenuto
    combined_text = f"{title} {content}"

    # Rimuove caratteri speciali e converte in minuscolo
    combined_text = re.sub(r"[’']", " ", combined_text)  # Rimuove ' e ’
    combined_text = re.sub(r"[^\w\s]", "", combined_text).lower()

    # Tokenizzazione e lemmatizzazione
    doc = nlp(combined_text)
    tokens = [
        token.lemma_ for token in doc
        if token.text not in combined_stopwords and not token.is_punct and not token.is_space
    ]
    # Ricostruisce il testo pulito
    return ' '.join(tokens)

# Applicazione sui contenuti
print('Applying cleaning function')
cleaned_contents_optimized = [
    clean_combined_text(
        content.get("title", {}).get("rendered", "").strip(),
        content.get("content", {}).get("rendered", "")
    )
    for content in all_contents
]

# Anteprima
print(cleaned_contents_optimized[:5])


# %%
# embeddings
print('generating embeddings')
# Scegli il modello

# La libreria SentenceTransformers supporta molti modelli ottimizzati per lingue multiple, incluso l'italiano. 
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
#Number of clusters (topics): 57
#Number of outliers: 78

# Modello multilingua di Facebook AI, eccellente per compiti NLP in italiano.
#model = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
#Number of clusters (topics): 55
#Number of outliers: 83

# Modello multilingua di Google, specificamente progettato per supportare più di 100 lingue.
#model = SentenceTransformer('sentence-transformers/LaBSE')
#Number of clusters (topics): 56
#Number of outliers: 80

print(f'model: {model}')
# Genera embeddings
embeddings = model.encode(cleaned_contents_optimized)

# %%
# Controlla la forma degli embeddings
print(f"Shape of embeddings: {len(embeddings)}, {len(embeddings[0])}")

# %%
# clustering con bertopic
print('clustering with bertopic')
# Inizializza il modello BERTopic
#topic_model = BERTopic(language="multilingual")  # Scegli "italian" per il modello linguistico di SpaCy
#topic_model = BERTopic(language="multilingual", min_topic_size=2)
#topic_model = BERTopic(language="italian", min_topic_size=2)
topic_model = BERTopic(language="italian", 
    min_topic_size=2,  # Slightly increase minimum cluster size
    #low_memory=True,  # For large datasets
    calculate_probabilities=True,
    #nr_topics='auto',  # Let algorithm determine optimal number
    verbose=True
)
# Adatta il modello usando gli embeddings
topics, probs = topic_model.fit_transform(cleaned_contents_optimized, embeddings)
# Mostra i cluster trovati
#print(topic_model.get_topic_info())

# STAMPA
unique_topics = set(topics) - {-1}
print(f"Number of clusters (topics): {len(unique_topics)}")
# Conta quanti contenuti sono outlier (topic -1)
outlier_count = topics.count(-1)
print(f"Number of outliers: {outlier_count}")
# Trova gli indici degli outliers (documenti con topic assegnato a -1)
outlier_indices = [i for i, t in enumerate(topics) if t == -1]
# Recupera gli URL o altri dettagli dai tuoi dati originali
outlier_urls = [all_contents[i].get('link', "URL non disponibile") for i in outlier_indices]
# Stampa il numero e la lista degli URL degli outliers
print("Lista degli URL degli outliers:")
for url in outlier_urls:
    print(url)

# %%
# Visualizza la mappa dei topic
#topic_model.visualize_topics()
fig = topic_model.visualize_topics()

# Salva la visualizzazione come file HTML
html_filename = "topics_map.html"
fig.write_html(html_filename)

# Apri il file HTML nel browser predefinito
webbrowser.open(html_filename)

# %%
# Mostra il contenuto di un topic specifico
topic_id = -1  # Cambia ID per esplorare altri topic
print(topic_model.get_topic(topic_id))


# %%
# Ottieni informazioni sui topic
topic_info = topic_model.get_topic_info()
topic_info

# %%
# Visualizza la distribuzione dei topic
#topic_model.visualize_barchart(top_n_topics=16)
fig = topic_model.visualize_barchart(top_n_topics=16)

# Salva la visualizzazione come file HTML
html_filename = "topics_barchart.html"
fig.write_html(html_filename)

# Apri il file HTML nel browser predefinito
webbrowser.open(html_filename)

# %%
# Visualizza le relazioni tra topic
#topic_model.visualize_hierarchy()

fig = topic_model.visualize_hierarchy()

# Salva la visualizzazione come file HTML
html_filename = "topics_hierarchy.html"
fig.write_html(html_filename)

# Apri il file HTML nel browser predefinito
webbrowser.open(html_filename)

# %%
# Visualizza i documenti per topic
#topic_model.visualize_documents(cleaned_contents_optimized, embeddings=embeddings)

fig = topic_model.visualize_documents(cleaned_contents_optimized, embeddings=embeddings)


# Salva la visualizzazione come file HTML
html_filename = "topics_docs.html"
fig.write_html(html_filename)

# Apri il file HTML nel browser predefinito
webbrowser.open(html_filename)
# %%
#topic_model.visualize_heatmap(top_n_topics=16)
#topic_model.visualize_heatmap()
fig = topic_model.visualize_heatmap()

# Salva la visualizzazione come file HTML
html_filename = "topics_hm.html"
fig.write_html(html_filename)

# Apri il file HTML nel browser predefinito
webbrowser.open(html_filename)


# %%
# Mappa contenuti raw con cluster e cleaned content
data = pd.DataFrame({
    "raw_content": [content.get('content', {}).get('rendered', "") for content in all_contents],
    "cleaned_content": cleaned_contents_optimized,
    "topic": topics
})

# %%
# Aggiungi una colonna con gli URL
data['url'] = [content.get('link', "") for content in all_contents]

# %%
# prime 3 kw per cluster
# Funzione per ottenere le prime 3 parole chiave di un topic
def get_topic_keywords(topic_id):
    topic = topic_model.get_topic(topic_id)
    if topic:  # Se il topic esiste
        return ", ".join([word for word, _ in topic[:3]])
    return "N/A"

# Aggiungi le parole chiave principali per ogni cluster
data['topic_keywords'] = data['topic'].apply(get_topic_keywords)

# %%
data.head()

# %%
# Funzione per estrarre tutti i link href da un contenuto
def extract_links(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    links = [a.get('href') for a in soup.find_all('a', href=True)]
    return links

# Aggiungi una colonna con i link estratti
data['links'] = data['raw_content'].apply(extract_links)

# %%
data.head()

# %%
# Funzione per verificare le interconnessioni
def check_cluster_links(cluster_id, cluster_data):
    # Filtra i contenuti del cluster
    cluster = cluster_data[cluster_data['topic'] == cluster_id]
    
    # Crea mappature di URL e link
    links_mapping = {row.Index: set(row.links) for row in cluster.itertuples()}
    urls_mapping = {row.Index: row.url for row in cluster.itertuples()}
    
    # Verifica delle interconnessioni
    missing_links = []
    for idx, links in links_mapping.items():
        # Verifica che ogni altro documento del cluster sia linkato
        for other_idx, other_url in urls_mapping.items():
            if idx != other_idx and other_url not in links:
                missing_links.append((urls_mapping[idx], other_url))  # Salva gli URL
    
    return missing_links

# %%
# Verifica per tutti i cluster
cluster_issues = {}
for cluster_id in data['topic'].unique():
    if cluster_id == -1:  # Salta il cluster degli outliers
        continue
    issues = check_cluster_links(cluster_id, data)
    cluster_issues[cluster_id] = issues

# %%
# Report finale leggibile
#for cluster_id, issues in cluster_issues.items():
#    topic_keywords = get_topic_keywords(cluster_id)
    
#    if not issues:
#        print(f"Tutti i contenuti del cluster '{topic_keywords}' sono interconnessi. Ottimo lavoro!")
#    else:
        #print(f"Il cluster '{topic_keywords}' ha problemi di interconnessione:")
#        for src_url, tgt_url in issues:
            #print(f" - Il contenuto {src_url} non linka il contenuto {tgt_url}")

# %%
# conteggio suggerimenti
# Calcola il numero totale di suggerimenti di interlinking
total_suggestions = sum(len(issues) for issues in cluster_issues.values())
print(f"Total suggestions for interlinking: {total_suggestions}")

# %%
# Rimuovi caratteri non validi per il nome del file
def clean_domain(domain):
    # Rimuovi "http:", "https:", "/", e sostituisci "." con "_"
    return re.sub(r'[^\w]', '_', domain.strip("http:").strip("https:").strip("/"))


# Genera il nome del file basato sul dominio
report_filename = f"{clean_domain(domain)}_interlinking_report.html"

# Prepara i dati per il report
report_data = []

for cluster_id, issues in cluster_issues.items():
    # Ottieni le parole chiave del topic
    topic_keywords = get_topic_keywords(cluster_id)  
    
    if not issues:
        # Tutti i contenuti sono interconnessi
        report_data.append({
            "Cluster": topic_keywords,
            "Status": "Tutti i contenuti sono interconnessi",
            "Source": None,
            "Target": None
        })
    else:
        # Contenuti non interconnessi
        for src_url, tgt_url in issues:
            report_data.append({
                "Cluster": topic_keywords,
                "Status": "Problema di interconnessione",
                "Source": src_url,
                "Target": tgt_url
            })

# Converti i dati in un DataFrame
report_df = pd.DataFrame(report_data)

# Esporta il report in formato HTML interattivo
html_output = report_df.to_html(
    index=False, 
    classes='table table-bordered table-hover', 
    escape=False
)

# Prepara una struttura HTML semplice e leggibile
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/4.6.2/css/bootstrap.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/4.6.2/js/bootstrap.bundle.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .accordion-header {{ 
            color: #333;
            cursor: pointer;
            padding: 15px;
            background-color: #f8f9fa;
            border: none;
            text-align: left;
            width: 100%;
            margin-bottom: 2px;
        }}
        .accordion-header:hover {{
            background-color: #e9ecef;
        }}
        .accordion-content {{
            padding: 15px;
        }}
        ul {{ margin-left: 20px; list-style-type: disc; }}
        li {{ margin-bottom: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Interlinking Report per il dominio {domain}</h1>
        <p><strong>Totale suggerimenti di interlinking:</strong> {total_suggestions}</p>

        <div class="accordion" id="reportAccordion">
"""

# Genera il contenuto per ogni cluster
for i, (cluster_id, issues) in enumerate(cluster_issues.items(), 1):
    topic_keywords = get_topic_keywords(cluster_id)
    html_content += f"""
            <div class="card">
                <div class="card-header" id="heading{i}">
                    <h2 class="mb-0">
                        <button class="accordion-header collapsed" type="button" 
                                data-toggle="collapse" 
                                data-target="#collapse{i}" 
                                aria-expanded="false" 
                                aria-controls="collapse{i}">
                            Cluster: {topic_keywords}
                        </button>
                    </h2>
                </div>
                <div id="collapse{i}" 
                     class="collapse" 
                     aria-labelledby="heading{i}" 
                     data-parent="#reportAccordion">
                    <div class="accordion-content">
    """
    
    if not issues:
        html_content += "<p><strong>Tutti i contenuti sono interconnessi. Ottimo lavoro!</strong></p>"
    else:
        html_content += "<ul>"
        for src_url, tgt_url in issues:
            html_content += f"<li>Il contenuto <a href='{src_url}' target='_blank'>{src_url}</a> non linka il contenuto <a href='{tgt_url}' target='_blank'>{tgt_url}</a></li>"
        html_content += "</ul>"
    
    html_content += """
                    </div>
                </div>
            </div>
    """

# Chiudi l'HTML
html_content += """
        </div>
    </div>
</body>
</html>
"""
# Salva il file HTML
with open(report_filename, "w", encoding="utf-8") as file:
    file.write(html_content)

print(f"Report statico salvato come '{report_filename}'")
# Apri automaticamente il report nel browser predefinito
# Ottieni il percorso assoluto del file
report_path = os.path.abspath(report_filename)

# Apri il file nel browser predefinito
webbrowser.open(f"file://{report_path}")
#webbrowser.open(report_filename)


# %%
