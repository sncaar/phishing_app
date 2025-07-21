
import re
import socket
import whois
import requests
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from bs4 import BeautifulSoup

# Modelimize girdi olarak kullanılan 87 öznitelik adı
FEATURE_NAMES = [
    'length_url','length_hostname','ip','nb_dots','nb_hyphens','nb_at',
    'nb_qm','nb_and','nb_or','nb_eq','nb_underscore','nb_tilde',
    'nb_percent','nb_slash','nb_star','nb_colon','nb_comma',
    'nb_semicolumn','nb_dollar','nb_space','nb_www','nb_com',
    'nb_dslash','http_in_path','https_token','ratio_digits_url',
    'ratio_digits_host','punycode','port','tld_in_path',
    'tld_in_subdomain','abnormal_subdomain','nb_subdomains',
    'prefix_suffix','random_domain','shortening_service',
    'path_extension','nb_redirection','nb_external_redirection',
    'length_words_raw','char_repeat','shortest_words_raw',
    'shortest_word_host','shortest_word_path','longest_words_raw',
    'longest_word_host','longest_word_path','avg_words_raw',
    'avg_word_host','avg_word_path','phish_hints','domain_in_brand',
    'brand_in_subdomain','brand_in_path','suspecious_tld',
    'statistical_report','nb_hyperlinks','ratio_intHyperlinks',
    'ratio_extHyperlinks','ratio_nullHyperlinks','nb_extCSS',
    'ratio_intRedirection','ratio_extRedirection',
    'ratio_intErrors','ratio_extErrors','login_form',
    'external_favicon','links_in_tags','submit_email',
    'ratio_intMedia','ratio_extMedia','sfh','iframe',
    'popup_window','safe_anchor','onmouseover','right_clic',
    'empty_title','domain_in_title','domain_with_copyright',
    'whois_registered_domain','domain_registration_length',
    'domain_age','web_traffic','dns_record','google_index',
    'page_rank'
]

def extract_features(url: str) -> pd.DataFrame:
    """
    Verilen URL için 87 öznitelik çıkarıp pandas.DataFrame(1×87) döner.
    """
    p = urlparse(url)
    host = p.netloc.lower()
    path = p.path or ""
    f = {}

    # — Basit sayaç & karakter sayıları —
    f['length_url']        = len(url)
    f['length_hostname']   = len(host)
    f['ip']                = int(bool(re.match(r'^\d+\.\d+\.\d+\.\d+$', host)))
    f['nb_dots']           = url.count('.')
    f['nb_hyphens']        = url.count('-')
    f['nb_at']             = url.count('@')
    f['nb_qm']             = url.count('?')
    f['nb_and']            = url.count('&')
    f['nb_or']             = url.count('|')
    f['nb_eq']             = url.count('=')
    f['nb_underscore']     = url.count('_')
    f['nb_tilde']          = url.count('~')
    f['nb_percent']        = url.count('%')
    f['nb_slash']          = url.count('/')
    f['nb_star']           = url.count('*')
    f['nb_colon']          = url.count(':')
    f['nb_comma']          = url.count(',')
    f['nb_semicolumn']     = url.count(';')
    f['nb_dollar']         = url.count('$')
    f['nb_space']          = url.count(' ')
    f['nb_www']            = url.lower().count('www')
    f['nb_com']            = host.count('.com')
    f['nb_dslash']         = url.count('//')
    f['http_in_path']      = int('http' in path)
    f['https_token']       = int('https' in host)
    f['ratio_digits_url']  = sum(c.isdigit() for c in url) / max(1, len(url))
    f['ratio_digits_host'] = sum(c.isdigit() for c in host) / max(1, len(host))
    f['punycode']          = int(host.startswith('xn--'))
    f['port']              = int(bool(p.port))

    # — TLD ve alt alan adı özellikleri —
    f['tld_in_path']       = int(any(t in path.lower() for t in ['.com','.net','.org']))
    sub = host.split('.')
    f['tld_in_subdomain']  = int(sub[0] in ['com','net','org'])
    f['abnormal_subdomain']= int(bool(re.match(r'^[0-9\-]+$', sub[0])))
    f['nb_subdomains']     = max(0, len(sub)-2)

    # — Domain isim karakteristiği & kısa URL tespiti —
    f['prefix_suffix']     = int('-' in host and host.count('-') > 1)
    f['random_domain']     = int(len(set(host)) > len(host)*0.8)
    shorteners = ['bit.ly','goo.gl','t.co','tinyurl.com']
    f['shortening_service']= int(any(s in host for s in shorteners))

    # — Dosya uzantısı & yönlendirme (redirect) —
    f['path_extension']          = int(bool(re.search(r'\.(php|html|aspx|jsp)$', path)))
    f['nb_redirection']          = url.count('redirect')
    f['nb_external_redirection'] = url.count('external')

    # — Kelime/karakter istatistikleri —
    words = re.findall(r'\w+', url)
    f['length_words_raw']   = sum(len(w) for w in words)
    f['char_repeat']        = max((words.count(w) for w in words), default=0)
    f['shortest_words_raw'] = min((len(w) for w in words), default=0)
    f['shortest_word_host'] = min((len(w) for w in host.split('.')), default=0)
    f['shortest_word_path'] = min((len(w) for w in path.split('/')), default=0)
    f['longest_words_raw']  = max((len(w) for w in words), default=0)
    f['longest_word_host']  = max((len(w) for w in host.split('.')), default=0)
    f['longest_word_path']  = max((len(w) for w in path.split('/')), default=0)
    f['avg_words_raw']      = np.mean([len(w) for w in words]) if words else 0
    f['avg_word_host']      = np.mean([len(w) for w in host.split('.')]) if sub else 0
    f['avg_word_path']      = np.mean([len(w) for w in path.split('/')]) if path else 0

    # — Phishing ipuçları & marka bazlı öznitelikler —
    hints  = ['secure','verify','login','update','account']
    brands = ['google','paypal','bankofamerica','github','amazon']
    f['phish_hints']        = int(any(h in path.lower() for h in hints))
    f['domain_in_brand']    = int(any(b in host for b in brands))
    f['brand_in_subdomain'] = int(any(b in sub[0] for b in brands))
    f['brand_in_path']      = int(any(b in path.lower() for b in brands))
    tlds   = ['info','xyz','top','online','tech']
    f['suspecious_tld']     = int(host.split('.')[-1] in tlds)

    # — Whois & domain yaşı öznitelikleri —
    try:
        w = whois.whois(host)
        f['whois_registered_domain']    = 1
        cdate = w.creation_date
        if isinstance(cdate, list): cdate = cdate[0]
        days = (pd.Timestamp('today') - pd.to_datetime(cdate)).days
        f['domain_age']                 = days
        f['domain_registration_length'] = days
    except:
        f['whois_registered_domain']    = 0
        f['domain_age']                 = 0
        f['domain_registration_length'] = 0

    # — Web traffic (dummy olarak 0 ayarlandı) —
    f['web_traffic'] = 0

    # — DNS kaydı kontrolü —
    try:
        socket.gethostbyname(host)
        f['dns_record'] = 1
    except:
        f['dns_record'] = 0

    # — Google index & page rank (dummy olarak 0) —
    f['google_index'] = 0
    f['page_rank']    = 0

    # — HTML içerik tabanlı öznitelikler —
    try:
        r = requests.get(url, timeout=3)
        soup = BeautifulSoup(r.text, 'html.parser')

        links = soup.find_all('a')
        f['nb_hyperlinks']       = len(links)
        int_h = sum(1 for a in links if a.get('href','').startswith('http') and host in a.get('href',''))
        ext_h = sum(1 for a in links if a.get('href','').startswith('http') and host not in a.get('href',''))
        null_h= len(links) - int_h - ext_h
        total = len(links) or 1
        f['ratio_intHyperlinks'] = int_h / total
        f['ratio_extHyperlinks'] = ext_h / total
        f['ratio_nullHyperlinks']= null_h / total

        css = [l for l in soup.find_all('link') if '.css' in l.get('href','')]
        f['nb_extCSS']           = len([l for l in css if l.get('href','').startswith('http') and host not in l.get('href','')])

        # Basit yönlendirme/error dummy atama
        f['ratio_intRedirection'] = 0
        f['ratio_extRedirection'] = 0
        f['ratio_intErrors']      = 0
        f['ratio_extErrors']      = 0

        forms = soup.find_all('form')
        f['login_form']      = int(any('password' in str(frm).lower() for frm in forms))
        fav = soup.find('link', rel=lambda x: x and 'icon' in x.lower())
        f['external_favicon']= int(bool(fav and fav.get('href','').startswith('http') and host not in fav.get('href','')))

        f['links_in_tags']   = len(links)
        f['submit_email']    = int(any('mailto:' in a.get('href','') for a in links))

        media = soup.find_all(['audio','video'])
        int_m = sum(1 for m in media if m.get('src','').startswith('http') and host in m.get('src',''))
        ext_m = sum(1 for m in media if m.get('src','').startswith('http') and host not in m.get('src',''))
        total_m = len(media) or 1
        f['ratio_intMedia']  = int_m / total_m
        f['ratio_extMedia']  = ext_m / total_m

        f['sfh']             = int(any(frm.get('action','').startswith('http') for frm in forms))
        f['iframe']          = len(soup.find_all('iframe'))
        f['popup_window']    = int(any('window.open' in script.text for script in soup.find_all('script')))
        f['safe_anchor']     = int(any('javascript:void' in a.get('href','') for a in links))
        f['onmouseover']     = int(any(tag.has_attr('onmouseover') for tag in soup.find_all()))
        f['right_clic']      = int(any(tag.has_attr('oncontextmenu') for tag in soup.find_all()))
        f['empty_title']     = int(soup.title is None or not soup.title.text.strip())
        f['domain_in_title'] = int(bool(soup.title and host in soup.title.text.lower()))
        f['domain_with_copyright'] = int(host in soup.get_text().lower())

    except:
        for k in [
            'nb_hyperlinks','ratio_intHyperlinks','ratio_extHyperlinks','ratio_nullHyperlinks',
            'nb_extCSS','ratio_intRedirection','ratio_extRedirection','ratio_intErrors','ratio_extErrors',
            'login_form','external_favicon','links_in_tags','submit_email','ratio_intMedia','ratio_extMedia',
            'sfh','iframe','popup_window','safe_anchor','onmouseover','right_clic',
            'empty_title','domain_in_title','domain_with_copyright'
        ]:
            f[k] = 0

    # Eksik kalan tüm öznitelikleri 0 ile doldur
    for name in FEATURE_NAMES:
        if name not in f:
            f[name] = 0

    return pd.DataFrame([f], columns=FEATURE_NAMES)