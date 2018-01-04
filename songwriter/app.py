from flask import Flask, request, jsonify, json
import songbird

app = Flask(__name__)

songbird.init()

@app.route('/')
def index():
    return 'Song Bird'

@app.route('/write')
@app.route('/write/<artist>')
def write(artist='thebeatles'):
    print(artist)
    sentence = request.args.get('text')
    print(sentence)
    song = songbird.gettext(sentence)
    return json.dumps({
        'artist': artist,
        'song': song
    })

@app.route('/artists')
def artists():
    return json.dumps([{
        'key': 'thebeatles',
        'value': 'The Beatles'
    }, {
        'key': 'thebeachboys',
        'value': 'The Beach Boys'
    }])

if __name__ == '__main__':
    app.run(debug=True, host='localhost')