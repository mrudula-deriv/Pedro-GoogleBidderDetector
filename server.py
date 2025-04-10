import subprocess
import aiohttp
import json
import asyncio
from playwright.async_api import async_playwright
import datetime
from datetime import timezone
import pytz
import urllib.parse
import logging
import os, re, requests, random
from urllib.parse import urlparse
from flask import Flask, jsonify, request
from openai import OpenAI
from dotenv import load_dotenv

# Semaphore to control concurrency
semaphore = asyncio.Semaphore(3)

KEYWORD_DELAY_MIN = 20
KEYWORD_DELAY_MAX = 60
COUNTRY_DELAY_MIN = 300
COUNTRY_DELAY_MAX = 600
INITIAL_VPN_DELAY = 10
VPN_RETRY_DELAY = 5
VPN_RETRY_ATTEMPTS = 3

# %%
'''with open('config/countries.json', 'r') as f:
    countries_data = json.load(f)
    COUNTRIES = {country['country']: country for country in countries_data['countries']}
with open('config/keywords.json', 'r') as f:
    keywords_data = json.load(f)
    KEYWORDS = [item['keyword'] for item in keywords_data]
'''

def load_config():
    """Load country and keyword data from JSON files."""
    countries, keywords = {}, []
    with open('config/countries.json', 'r') as f:
        countries_data = json.load(f)
        countries = {country['country']: country for country in countries_data['countries']}
    with open('config/keywords.json', 'r') as f:
        keywords_data = json.load(f)
        keywords = [item['keyword'] for item in keywords_data]
    logging.info("Configuration files loaded successfully.")
    return countries, keywords

COUNTRIES, KEYWORDS = load_config()

# %%
load_dotenv(dotenv_path=".env.local")
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")
from supabase import create_client
from datetime import datetime

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)


# %%
def connect_vpn(country):
    """
    Connect to a VPN for the specified country using the expresso command-line tool.
    
    Args:
        country (str): The country to connect to.
    """
    # Get the country codes
    country_data = COUNTRIES[country]
    
    # Randomly select a code if multiple codes exist, otherwise use glcode
    if isinstance(country_data["code"], list) and len(country_data["code"]) > 1:
        selected_code = random.choice(country_data["code"])
        print(f"Multiple codes available for {country}. Randomly selected: {selected_code}")
    else:
        selected_code = country_data["glcode"]
        print(f"Using glcode for {country}: {selected_code}")
    
    try:
        # Check the current VPN status
        status_command = ["expresso", "status"]
        status_result = subprocess.run(status_command, capture_output=True, text=True, check=True)
        
        # Parse the status output to determine if connected
        status_output = status_result.stdout.strip()
        
        # Check if the output indicates that we are connected
        if "VPN connected" in status_output:
            # Extract the current location from the status output
            current_location = status_output.split("'")[1].replace("'", "").split(" (")[0].split(" -")[0].strip()
            print(f"Currently connected to: {current_location}")
            if country.lower() in current_location.lower():
                print(f"Already connected to {country}. No action needed.")
                return current_location
            
            # Change the location to the specified country
            else:
                print(f"Changing VPN location from {current_location} to {country} (code: {selected_code}).")
                connect_command = ["expresso", "connect", "--change", selected_code]
                
        else:
            # Not connected, connect to the specified country
            print(f"Not connected. Connecting to {country} (code: {selected_code}).")
            connect_command = ["expresso", "connect", selected_code]
            
        # Execute the connect command
        connect_result = subprocess.run(connect_command, capture_output=True, text=True, check=True)
        print("VPN connection command output:", connect_result.stdout.strip())
        verify_status = subprocess.run(status_command, capture_output=True, text=True, check=True)
        final_location = verify_status.stdout.strip().split("'")[1].replace("'", "").split(" (")[0].split(" -")[0].strip()
        return final_location
        
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.stderr.strip()}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

def disconnect_vpn():
    """
    Disconnect from the current VPN connection using expresso command-line tool.
    Returns a dictionary with success status and message.
    """
    try:
        # First check if VPN is connected
        status_command = ["expresso", "status"]
        status_result = subprocess.run(status_command, capture_output=True, text=True)
        
        if "VPN connected" not in status_result.stdout:
            print("VPN is already disconnected.")
            return {
                "success": True,
                "message": "VPN already disconnected"
            }
        
        # Run disconnect command
        disconnect_command = ["expresso", "disconnect"]
        result = subprocess.run(disconnect_command, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("VPN disconnected successfully.")
            return {
                "success": True,
                "message": "VPN disconnected successfully"
            }
        else:
            print(f"Error disconnecting VPN: {result.stderr}")
            return {
                "success": False,
                "message": f"Error: {result.stderr}"
            }
            
    except subprocess.CalledProcessError as e:
        error_msg = f"Command error: {e.stderr}"
        print(error_msg)
        return {
            "success": False,
            "message": error_msg
        }
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(error_msg)
        return {
            "success": False,
            "message": error_msg
        }


# %%
#Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MAX_RETRIES = 8
#REGEX_SEARCH_PATTERN = r'affiliate'


# %%
def analyze_ad_and_url(ad, url=None):
    try:
        url_to_analyze = url or ad.get('link', '')
        
        prompt = f"""
        Analyze this ad content and URL for Deriv affiliate patterns:
        
        URL: {url_to_analyze}
        Title: {ad.get('title', '')}
        Displayed Link: {ad.get('displayed_link', '')}
        
        Check for:
        1. "deriv.com" anywhere in the content
        2. MyAffiliates pattern: "affiliate_" followed by numbers
        3. DynamicWorks pattern: "utm_source=cu" or "utm_source=c" followed by numbers
        
        Return a JSON object with:
        1. contains_deriv (boolean)
        2. affiliate_type (string: "MyAffiliates" or "DynamicWorks" or null)
        3. affiliate_id (number or null)
        4. found_in (array of where patterns were found: "url", "title", "displayed_link")
        5. flag (number: 1 if both contains_deriv AND affiliate_id are true, 0 otherwise)
        """
        
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": "You are an ad analysis assistant. Analyze URLs and content for specific patterns."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )
        
        # Parse AI response
        analysis = json.loads(response.choices[0].message.content)
        
        # Ensure flag is set correctly based on contains_deriv and affiliate_id
        analysis['flag'] = 1 if analysis.get('contains_deriv') and analysis.get('affiliate_id') else 0
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error in ad analysis: {str(e)}")
        return {
            "contains_deriv": False,
            "affiliate_type": None,
            "affiliate_id": None,
            "found_in": [],
            "flag": 0,
            "error": str(e)
        }


# %%
async def follow_redirects(url, session=None):
    """
    Follow redirects for a given URL and track the redirect chain.
    
    Args:
        url (str): The initial URL to follow
        session (aiohttp.ClientSession, optional): The aiohttp client session.
                                                If None, a new session is created.
        
    Returns:
        dict: Dictionary containing:
            - final_url: The final landing URL after following all redirects
            - redirect_chain: List of all URLs in the redirect chain
    """
    logger.info(f"Following redirects for URL: {url}")
    
    # Initialize result dictionary
    result = {
        "initial_url": url,
        "final_url": url,  # Default to original URL in case of error
        "redirect_chain": [url],
        "redirect_count": 0
    }
    
    try:
        # Set headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Follow redirects manually to track the chain
        #session = requests.Session() # Removed requests.Session()
        async with aiohttp.ClientSession() as local_session:
            _session = session or local_session
            current_url = url
            
            # Limit the number of redirects to prevent infinite loops
            max_redirects = 20
            while result['redirect_count'] < max_redirects:
                try:
                    async with _session.get(
                        current_url, 
                        headers=headers, 
                        allow_redirects=False,
                        timeout=10
                    ) as response:
                        # Add current URL to the chain if it's not already there
                        if current_url not in result['redirect_chain']:
                            result['redirect_chain'].append(current_url)
                        
                        # Check if we've reached the final URL (no more redirects)
                        if response.status < 300 or response.status >= 400 or 'Location' not in response.headers:
                            break
                            
                        # Get the next URL in the chain
                        next_url = response.headers['Location']
                        
                        # Handle relative URLs
                        if next_url.startswith('/'):
                            parsed_url = urlparse(current_url)
                            next_url = f"{parsed_url.scheme}://{parsed_url.netloc}{next_url}"
                            
                        current_url = next_url
                        result['redirect_count'] += 1
                        
                except aiohttp.ClientError as e: # Catch aiohttp exceptions
                    logger.error(f"Error following redirect for {current_url}: {e}")
                    break
            
            # Make a final request with allow_redirects=True to get the final URL
            try:
                async with _session.get(current_url, headers=headers, allow_redirects=True, timeout=10) as final_response:
                    result['final_url'] = str(final_response.url)
                    
                    # Add final URL to chain if not already there
                    if result['final_url'] not in result['redirect_chain']:
                        result['redirect_chain'].append(result['final_url'])
                    
            except aiohttp.ClientError as e: # Catch aiohttp exceptions
                # If final request fails, use the last URL we successfully reached
                result['final_url'] = current_url
                logger.error(f"Error on final request: {e}")
            
            logger.info(f"Redirect chain completed. Final URL: {result['final_url']}")
            return result
        
    except Exception as e:
        logger.error(f"Error in follow_redirects: {e}")
        return result


# %%
async def google_search(q, timestamp_str=None, session=None, retries=0):
    """
    Perform a Google search and extract ads using Playwright
    
    Args:
        q (str): The search query
        timestamp_str (str): Timestamp for file naming
        session (aiohttp.ClientSession, optional): The aiohttp client session.
                                                If None, a new session is created.
        retries (int): Current retry attempt
        
    Returns:
        tuple: (results dict, screenshot path, html content)
    """
    if retries >= MAX_RETRIES:
        logger.error("Maximum retries reached. CAPTCHA is blocking our attempts.")
        return None, None, None

    search_term = urllib.parse.quote_plus(q)
    #country_code = get_country_code(country)
    #connect_vpn(country)
    search_url = f"https://www.google.com/search?q={search_term}"

    async with async_playwright() as pw:
        screen_width = random.choice([1366, 1920, 1600, 1440])
        screen_height = random.choice([768, 1080, 900, 900])

        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/537.36",
        ]
        user_agent = random.choice(user_agents)

        browser = await pw.chromium.launch(headless=True, args=[
            "--start-maximized",
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-infobars",
            "--disable-dev-shm-usage"
        ])
        context = await browser.new_context(
            viewport={"width": screen_width, "height": screen_height},
            user_agent=user_agent
        )

        page = await context.new_page()
        
        try:
            await page.goto(search_url, wait_until="domcontentloaded")

            # 🚀 **Detect if CAPTCHA Exists**
            captcha_detected = await page.evaluate("""
                () => {
                    return document.body.innerText.includes("unusual traffic") ||
                           document.querySelector("#recaptcha") !== null ||
                           document.querySelector('form[action*="captcha"]') !== null;
                }
            """)

            if captcha_detected:
                logger.warning(f"CAPTCHA detected! Retrying... Retry {retries + 1}/{MAX_RETRIES}")
                await browser.close()
                await asyncio.sleep(random.uniform(3, 7))
                return await google_search(q, timestamp_str, session, retries + 1)  # Pass session

            await page.evaluate("""
                () => {
                    Object.defineProperty(navigator, 'webdriver', {get: () => false});
                }
            """)

            await page.wait_for_selector("h3", timeout=10000)

            # ✅ Take a screenshot of Google Search for debugging
            # Create logs directory if it doesn't exist
            os.makedirs("logs", exist_ok=True)
            # Use the provided timestamp or generate a new one
            file_timestamp = timestamp_str if timestamp_str else datetime.datetime.now(pytz.UTC).strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"logs/google_search_{file_timestamp}_try_{retries+1}.png"
            await page.screenshot(path=screenshot_path, full_page=True)
            
            # Get the HTML content of the page (will be saved later if needed)
            html_content = await page.content()

            # ✅ Extract Regular Search Results (Organic)
            search_results = await page.locator("a:has(h3)").evaluate_all(
                "nodes => nodes.map(n => ({ link: n.href, title: n.innerText }))"
            )

            # 🎯 **Extract Sponsored Ads with SITELINKS**
            sponsored_ads = await page.evaluate("""
                () => {
                    let adsData = [];
                    document.querySelectorAll('.uEierd').forEach(adBlock => {
                        let ad = {};
                        
                        let titleElement = adBlock.querySelector('h3') || adBlock.querySelector('.cfxYMc') || adBlock.querySelector('[role="heading"]');
                        let linkElement = adBlock.querySelector('a');
                        let displayedLinkElement = adBlock.querySelector('.x2VHCd');
                        let descriptionElement = adBlock.querySelector('.MUxGbd.yDYNvb');

                        // ✅ Capture Main Ad Data
                        if (titleElement) { ad.title = titleElement.innerText; }
                        
                        // Get the displayed link (what user sees)
                        if (displayedLinkElement) { ad.displayed_link = displayedLinkElement.innerText; }
                        
                        // Get the actual tracking URL (what user gets when right-clicking "Copy link address")
                        if (linkElement) {
                            // First try to get the data-rw attribute which contains the Google tracking URL
                            if (linkElement.getAttribute('data-rw')) {
                                ad.link = linkElement.getAttribute('data-rw');
                            } else if (linkElement.href) {
                                ad.link = linkElement.href;
                            }
                        }
                        
                        if (descriptionElement) { ad.description = descriptionElement.innerText; }

                        // ✅ Capture Sitelinks (Extra Ad Links)
                        let siteLinks = [];
                        adBlock.querySelectorAll('div[role="link"] a').forEach(link => {
                            let linkText = link.innerText.trim();
                            let sitelink = { "title": linkText };
                            
                            // Try to get the data-rw attribute for sitelinks too
                            if (link.getAttribute('data-rw')) {
                                sitelink.link = link.getAttribute('data-rw');
                            } else if (link.href) {
                                sitelink.link = link.href;
                            }
                            
                            if (linkText && sitelink.link) {
                                siteLinks.push(sitelink);
                            }
                        });

                        if (siteLinks.length > 0) { ad.sitelinks = siteLinks; }

                        adsData.push(ad);
                    });
                    return adsData;
                }
            """)

            await browser.close()

            # ✅ Save results to a JSON file
            results = {"search_results": search_results, "ads": sponsored_ads}
            #with open("google_results.json", "w") as f:
            #    json.dump(results, f, indent=4)

            return results, screenshot_path, html_content

        except Exception as e:
            logger.error(f"Error occurred during Google search: {e}")
            await browser.close()
            return await google_search(q, timestamp_str, session, retries + 1)  # Pass session



# %%
#import asyncio
#from asyncio import get_event_loop

async def get_ads_for_keyword(keyword, session=None):  # session is now a parameter
    """
    Get Google ads for a specified keyword using Playwright
    
    Args:
        keyword (str): The search term to find ads for
        session (aiohttp.ClientSession, optional): The aiohttp client session. 
                                                If None, a new session is created.
        
    Returns:
        dict: Google ads results or error message
    """
    try:
        # Validate keyword
        if keyword is None:
            logger.error("Missing keyword")
            return {"error": "Missing keyword"}
        
        logger.info(f"Processing search request for keyword: {keyword}")
        
        # Generate timestamp
        timestamp_utc = datetime.now(pytz.UTC)
        timestamp_str = timestamp_utc.strftime("%Y%m%d_%H%M%S")
        
        # Perform search (now awaiting the coroutine)
        # Ensure google_search() uses the session
        results, screenshot_path, html_content = await google_search(keyword, timestamp_str, session=session)  
        
        if results is None:
            return {"error": "Failed to get search results, possibly due to CAPTCHA"}
            
        # Process ads if found
        if results.get("ads", []):
            for ad in results['ads']:
                # Add basic info
                ad['keyword'] = keyword
                ad['timestamp'] = str(timestamp_utc)
                
                # Get final URL after redirects
                # Ensure follow_redirects() uses the session
                redirect_result = await follow_redirects(ad['link'], session=session)  
                ad['final_link'] = redirect_result['final_url']
                ad['redirect_chain'] = redirect_result['redirect_chain']
                
                # Use OpenAI to analyze the ad and final URL
                analysis = analyze_ad_and_url(ad, ad['final_link'])  # Assuming this doesn't need a session
                
                # Set flag using same logic as original
                ad['flag'] = 1 if analysis['contains_deriv'] and analysis['affiliate_id'] else 0
                
                # Only add affiliate info if flag is 1
                if ad['flag'] == 1:
                    ad['affiliate_id'] = analysis['affiliate_id']
                    ad['affiliate_type'] = analysis['affiliate_type']
            
            # Save HTML if affiliate ad found
            has_flag_one = any(ad.get('flag') == 1 for ad in results['ads'])
            if has_flag_one and html_content:
                os.makedirs("logs", exist_ok=True)
                html_path = f"logs/google_search_{timestamp_str}_with_affiliate.html"
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                logger.info(f"HTML content saved to {html_path} - affiliate ad found")
            
            logger.info(f"Returning {len(results['ads'])} ads")
            return results['ads']
        else:
            logger.info(f"No ads found for keyword '{keyword}'")
            return {
                "message": "No ads found for this keyword",
                "keyword": keyword
            }
            
    except Exception as e:
        logger.error(f"Error processing request for keyword '{keyword}': {e}")
        return {"error": str(e)}



# %%
async def save_to_supabase(ads_data, country, session=None):
    """
    Save ads data to Supabase only if flag=1
    
    Args:
        ads_data (list/dict): The ads data to save
        country (str): The VPN country used for the search
        session (aiohttp.ClientSession, optional): The aiohttp client session (if needed for Supabase)
    """
    try:
        if isinstance(ads_data, dict) and 'message' in ads_data:
            # No ads found - don't save to database
            print(f"No ads found for keyword: {ads_data['keyword']} in {country}")
            return None
            
        # Filter and save only ads with flag=1
        affiliate_ads = [ad for ad in ads_data if ad.get('flag') == 1]
        
        if not affiliate_ads:
            print(f"No affiliate ads found for {country}")
            return None
            
        # Process affiliate ads
        for ad in affiliate_ads:
            ad_data = {
                'keyword': ad['keyword'],
                'country': country,
                'timestamp': ad['timestamp'],
                'title': ad.get('title'),
                'displayed_link': ad.get('displayed_link'),
                'link': ad['link'],
                'final_link': ad['final_link'],
                'redirect_chain': json.dumps(ad['redirect_chain']),
                'flag': 1,  # We know it's 1 since we filtered
                'affiliate_id': str(ad.get('affiliate_id')),
                'affiliate_type': ad.get('affiliate_type')
            }
            
            # Example: If Supabase has an async client that needs a session
            #if session:
            #    result = await supabase.table('affiliate_links').insert(ad_data).execute(session=session)
            #else:
            result = supabase.table('affiliate_links').insert(ad_data).execute()
            
            print(f"Saved affiliate ad for {country} - ID: {ad_data['affiliate_id']}, Type: {ad_data['affiliate_type']}")
            
    except Exception as e:
        print(f"Error saving to Supabase: {str(e)}")
        raise


# %%
#keyword = "deriv"
#country = "Brazil"


# %%
async def test_search(keyword, country, session=None): # Pass session if using aiohttp
    """Test function to get ads and save to Supabase"""
    async with semaphore:
        try:
            logging.info(f"Searching ads for {keyword} in {country}")
            result = await get_ads_for_keyword(keyword, session) # Pass session
            if isinstance(result, dict) and 'error' in result:
                logging.error(f"Error getting ads for {keyword} in {country}: {result['error']}")
                return False
            logging.info(f"Found {len(result) if isinstance(result, list) else 0} ads for {keyword} in {country}")
            await save_to_supabase(result, country, session) # Pass session
            return True
        except Exception as e:
            logging.error(f"Error in test_search for {keyword} in {country}: {str(e)}")
            return False

async def run_searches(session=None):
    for country in COUNTRIES:
        vpn_connected = False
        location = None  # Initialize location variable
        
        for attempt in range(VPN_RETRY_ATTEMPTS):
            try:
                location = connect_vpn(country)  # Store the location
                if location:
                    vpn_connected = True
                    break
                else:
                    logging.warning(f"VPN connection failed for {country} (attempt {attempt + 1})")
                    await asyncio.sleep(VPN_RETRY_DELAY)
            except Exception as e:
                logging.error(f"Exception during VPN connection for {country}: {e}")
                await asyncio.sleep(VPN_RETRY_DELAY)
        if not vpn_connected:
            logging.error(f"Failed to connect to VPN for {country}, skipping")
            continue

        await asyncio.sleep(INITIAL_VPN_DELAY)
        logging.info(f"Connected to VPN for country: {location}")

        tasks = []
        for keyword in KEYWORDS:
            tasks.append(asyncio.create_task(test_search(keyword, location, session)))
            await asyncio.sleep(random.uniform(KEYWORD_DELAY_MIN, KEYWORD_DELAY_MAX))
        results = await asyncio.gather(*tasks)

        if all(results):
            logging.info(f"Successfully processed all keywords for {country}")
        else:
            logging.warning(f"Some keywords failed for {country}")

        disconnect_vpn()
        logging.info(f"Disconnected from VPN for country: {country}")
        await asyncio.sleep(random.uniform(COUNTRY_DELAY_MIN, COUNTRY_DELAY_MAX))

if __name__ == "__main__":
    async def main():
        async with aiohttp.ClientSession() as session:
            await run_searches(session)
    
    # Add this line to actually run the async main function
    asyncio.run(main())