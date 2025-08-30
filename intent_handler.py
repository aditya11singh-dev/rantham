INTENTS = [
    {
        "intent": "greeting",
        "keywords": ["hello", "hi", "namaste", "hey"],
        "response": "👋 Hello! I'm here to assist you with anything related to Dhonk Craft."
    },
    {
        "intent": "good_morning",
        "keywords": ["good morning", "morning"],
        "response": "🌞 Good Morning! Wishing you a beautiful day ahead. How can I help you today with Dhonk Craft?"
    },
    {
        "intent": "good_evening",
        "keywords": ["good evening", "evening"],
        "response": "🌇 Good Evening! Hope you had a great day. Let me know how I can help you with Dhonk Craft."
    },
    {
        "intent": "good_night",
        "keywords": ["good night", "night"],
        "response": "🌙 Good Night! Sweet dreams from Dhonk Craft. Let me know if you have any questions before you sleep!"
    },
    {
        "intent": "about",
        "keywords": ["about dhonk", "who are you", "what is dhonk", "intro"],
        "response": "🧵 Dhonk Craft is a sustainable clothing and craft brand that supports local artisans near Ranthambore National Park. It is led by Divya Khandal (Founder) and works with women artisans to promote eco-friendly fashion."
    },
    {
        "intent": "contact",
        "keywords": ["contact", "phone", "email", "mobile", "reach", "connect"],
        "response": """📞 <b>Contact Information:</b><br>
👩‍💼 Founder: Divya Khandal<br>
📧 Email: divz333@gmail.com<br>
📱 Phone: 9166167005<br><br>
👨‍💼 General Manager: Mr. Maan Singh<br>
📧 Email: mansinghr4@gmail.com<br>
📱 Phone: 9829854896"""
    },
    {
        "intent": "address",
        "keywords": ["address", "location", "where is dhonk", "map", "shop"],
        "response": """🏠 <b>Dhonk Craft</b><br>
📍 Near Ranthambore National Park, Village Sherpur, Sawai Madhopur, Rajasthan<br><br>
<img src="https://maps.googleapis.com/maps/api/staticmap?center=Dhonk+Craft,+Sawai+Madhopur&zoom=15&size=300x200&maptype=roadmap&markers=color:red%7Clabel:D%7CDhonk+Craft,+Sawai+Madhopur" style="max-width:100%; border-radius:10px;" /><br><br>
🔗 <a href='https://www.google.com/maps?q=Dhonk+Craft,+Sherpur,+Sawai+Madhopur' target='_blank'>Open in Google Maps</a>"""
    },
    {
        "intent": "thanks",
        "keywords": ["thank", "shukriya", "thanks a lot"],
        "response": "😊 You're welcome! Let me know if you have more questions about Dhonk Craft."
    },
    {
        "intent": "products",
        "keywords": ["show all products", "see products", "list items", "clothes", "crafts"],
        "response": "🧶 Here’s a list of some amazing Dhonk Craft products:<br>• Handcrafted Kurtis<br>• Eco-Friendly Bags<br>• Block-Printed Scarves<br>• Elephant Toys<br><br>🔗 <a href='https://dhonk.com/shop' target='_blank'>View Full Catalog</a>"
    },
    {
        "intent": "order_status",
        "keywords": ["order status", "where is my order", "track order", "shipment", "order update"],
        "response": "📦 Please share your Order ID to check the status. You can also email us at <b>divz333@gmail.com</b> for personalized help."
    },
    {
        "intent": "cod_info",
        "keywords": ["cash on delivery", "cod", "pay on delivery"],
        "response": "💵 Yes! We do offer <b>Cash on Delivery (COD)</b> across India. No extra charges!"
    },
    {
        "intent": "return_policy",
        "keywords": ["return", "exchange", "refund", "return policy"],
        "response": "🔄 You can return or exchange products within <b>7 days</b> of delivery.<br>Items must be unused and in original condition.<br>Email us at <b>divz333@gmail.com</b> to initiate a return."
    },
    {
        "intent": "menu",
        "keywords": [
            "what can you do", "services", "help", "how can you help", "assist", "options",
            "what do you offer", "what can do you for me", "tum mere liye kya kar sakte ho ,what can do you for me"
        ],
        "response": """🤖 I can help you with the following:<br><br>
<button onclick="sendQuick('About Dhonk Craft')">🔸 About Dhonk Craft</button><br>
<button onclick="sendQuick('Show all products')">🔸 Show All Products</button><br>
<button onclick="sendQuick('Order Status')">🔸 Check Order Status</button><br>
<button onclick="sendQuick('Cash on Delivery')">🔸 Cash on Delivery Info</button><br>
<button onclick="sendQuick('Return Policy')">🔸 Return & Exchange Policy</button><br>
<button onclick="sendQuick('Contact')">🔸 Contact Information</button><br>
<button onclick="sendQuick('Address')">🔸 Location & Address</button><br><br>
💡 Just click a button above or ask your question directly!"""
    }
]


def detect_intent(message):
    message = message.lower()
    for intent in INTENTS:
        if any(keyword in message for keyword in intent["keywords"]):
            return intent["intent"]
    return None


def get_intent_response(intent_name):
    for intent in INTENTS:
        if intent["intent"] == intent_name:
            return intent["response"]
    return None
