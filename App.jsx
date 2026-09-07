import Navbar from "./components/Navbar"
import Hero from "./components/Hero"
import Features from "./components/Features"
import Contact from "./components/Contact"
import Footer from "./components/Footer"

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <Hero />
      <Features />
      {/* yahan pkl aane ke baad Upload/Analyze section daalenge, id="demo" ke saath */}
      <Contact />
      <Footer />
    </div>
  )
}

export default App
