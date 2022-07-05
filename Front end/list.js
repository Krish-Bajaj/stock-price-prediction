const tickerDiv = document.getElementById('ticker-div')
const lastDayDiv = document.getElementById('last-day-div')
const nextDayDiv = document.getElementById('next-day-div')
const accuracyDiv = document.getElementById('accuracy-div')

tickerDiv.style.display = "none"
lastDayDiv.style.display = "none"
nextDayDiv.style.display = "none"
accuracyDiv.style.display = "none"

const tickerOutput = document.getElementById('ticker')
const lastDayOutput = document.getElementById('last-day')
const nextDayOutput = document.getElementById('next-day')
const accuracy = document.getElementById('accuracy')

function getTicker(symbol) {
    // e.preventDefault()
    tickerDiv.style.display = "flex"
    lastDayDiv.style.display = "flex"
    nextDayDiv.style.display = "flex"
    accuracyDiv.style.display = "flex"
    tickerOutput.innerText = symbol

    let query = {
        query: symbol
    }
    fetch('http://127.0.0.1:5000/', {
        method: 'POST',
        body: JSON.stringify(query),
        headers: {
            'Content-type': 'application/json; charset=UTF-8'
        }
    })
    .then(response => response.json())
    .then(json => {
        console.log(json)
        // document.getElementById("demo").innerHTML = json;
    })

    const response = {
        last: 123,
        next: 345,
        accuracy: '98%'
    }
    lastDayOutput.innerText = response['last']
    nextDayOutput.innerText = response['next']
    accuracy.innerText = response['accuracy']
}
